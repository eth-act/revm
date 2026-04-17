// Injectable U256 backend for revm.
//
// The `U256` type stores four `u64` limbs in little-endian word order
// (limbs[0] = least-significant word).  All operator overloads delegate at
// runtime to the globally-installed `Uint256Ops` backend.  Methods that must
// be `const fn` (constructors, byte conversions, const arithmetic) always use
// ruint via `alloy_primitives::U256` directly, bypassing the backend.
//
// Vendors install a custom backend once at startup via
// `install_uint256_backend()`.  The default backend (`DefaultUint256Ops`) is a
// thin wrapper around ruint.

use crate::OnceLock;

// ---------------------------------------------------------------------------
// Private helper: lift [u64; 4] → alloy_primitives::U256 for ruint calls.
// ---------------------------------------------------------------------------

#[inline(always)]
const fn ru(limbs: [u64; 4]) -> alloy_primitives::U256 {
    alloy_primitives::U256::from_limbs(limbs)
}

// ---------------------------------------------------------------------------
// Uint256Ops trait — the backend contract
// ---------------------------------------------------------------------------

/// Trait that a U256 arithmetic backend must implement.
///
/// Every method has a default implementation backed by ruint
/// (`alloy_primitives::U256`).  Vendors override only the operations they want
/// to accelerate or replace.
///
/// All values cross the boundary as `[u64; 4]` in little-endian limb order.
pub trait Uint256Ops: Send + Sync + core::fmt::Debug {
    // Only operations that are overridden by a custom backend are listed here.
    // Operations absent from this trait bypass dynamic dispatch entirely —
    // their `U256` methods call ruint directly.

    // ── Overflowing arithmetic ───────────────────────────────────────

    /// Wrapping multiplication with overflow flag. Returns `(result, overflowed)`.
    fn overflowing_mul(&self, a: [u64; 4], b: [u64; 4]) -> ([u64; 4], bool) {
        let (v, o) = ru(a).overflowing_mul(ru(b));
        (*v.as_limbs(), o)
    }

    // ── Wrapping arithmetic ──────────────────────────────────────────

    /// Wrapping division. Panics if `b` is zero.
    fn wrapping_div(&self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        *ru(a).wrapping_div(ru(b)).as_limbs()
    }

    /// Wrapping remainder. Panics if `b` is zero.
    fn wrapping_rem(&self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        *ru(a).wrapping_rem(ru(b)).as_limbs()
    }

    // ── Saturating arithmetic ────────────────────────────────────────

    /// Saturating multiplication. Saturates at `U256::MAX` on overflow.
    fn saturating_mul(&self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        *ru(a).saturating_mul(ru(b)).as_limbs()
    }

    // ── Checked arithmetic ───────────────────────────────────────────

    /// Checked multiplication. Returns `None` on overflow.
    fn checked_mul(&self, a: [u64; 4], b: [u64; 4]) -> Option<[u64; 4]> {
        ru(a).checked_mul(ru(b)).map(|v| *v.as_limbs())
    }

    /// Checked division. Returns `None` if `b` is zero.
    fn checked_div(&self, a: [u64; 4], b: [u64; 4]) -> Option<[u64; 4]> {
        ru(a).checked_div(ru(b)).map(|v| *v.as_limbs())
    }

    // ── Division / remainder ─────────────────────────────────────────

    /// Integer division. Panics if `b` is zero.
    fn div(&self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        *(ru(a) / ru(b)).as_limbs()
    }

    /// Integer remainder. Panics if `b` is zero.
    fn rem(&self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        *(ru(a) % ru(b)).as_limbs()
    }

    // ── EVM-specific exponential arithmetic ─────────────────────────

    /// Exponentiation: `base ^ exp`.
    fn pow(&self, base: [u64; 4], exp: [u64; 4]) -> [u64; 4] {
        *ru(base).pow(ru(exp)).as_limbs()
    }
}

// ---------------------------------------------------------------------------
// Default backend
// ---------------------------------------------------------------------------

/// Default U256 backend: all operations use ruint via `alloy_primitives`.
#[derive(Debug)]
pub struct DefaultUint256Ops;

impl Uint256Ops for DefaultUint256Ops {}

// ---------------------------------------------------------------------------
// Global injectable backend
// ---------------------------------------------------------------------------

static U256_OPS: OnceLock<std::boxed::Box<dyn Uint256Ops>> = OnceLock::new();

/// Install a custom U256 arithmetic backend globally.
///
/// Returns `true` if the backend was installed, `false` if one was already set.
/// Call this once at program startup, before any U256 arithmetic is performed.
pub fn install_uint256_backend<B: Uint256Ops + 'static>(backend: B) -> bool {
    U256_OPS.set(std::boxed::Box::new(backend)).is_ok()
}

/// Return the installed backend, lazily initialising the default if needed.
#[inline]
fn ops() -> &'static dyn Uint256Ops {
    U256_OPS
        .get_or_init(|| std::boxed::Box::new(DefaultUint256Ops))
        .as_ref()
}

// ---------------------------------------------------------------------------
// U256
// ---------------------------------------------------------------------------

/// 256-bit unsigned integer with an injectable arithmetic backend.
///
/// Stored as four `u64` limbs in little-endian word order (`limbs[0]` is the
/// least-significant word).  All operator overloads delegate to the installed
/// backend at runtime.  `const fn` methods always use ruint directly.
#[repr(transparent)]
#[derive(Copy, Clone, Default)]
pub struct U256([u64; 4]);

// PartialEq / Eq: elementwise on the raw limbs.
impl PartialEq for U256 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for U256 {}

// Ord: compare from the most-significant limb (index 3) downward.
impl Ord for U256 {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0[3]
            .cmp(&other.0[3])
            .then_with(|| self.0[2].cmp(&other.0[2]))
            .then_with(|| self.0[1].cmp(&other.0[1]))
            .then_with(|| self.0[0].cmp(&other.0[0]))
    }
}

impl PartialOrd for U256 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// Hash: hash the raw limb array.
impl core::hash::Hash for U256 {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

// Serde: wire-compatible with alloy_primitives::U256.
#[cfg(feature = "serde")]
impl serde::Serialize for U256 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ru(self.0).serialize(serializer)
    }
}
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for U256 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        alloy_primitives::U256::deserialize(deserializer).map(|v| Self(*v.as_limbs()))
    }
}

// ---------------------------------------------------------------------------
// Constants & constructors
// ---------------------------------------------------------------------------

impl U256 {
    /// Number of bits in this integer type.
    pub const BITS: usize = 256;

    /// Additive identity.
    pub const ZERO: Self = Self::from_limbs([0; 4]);

    /// Multiplicative identity.
    pub const ONE: Self = Self::from_limbs([1, 0, 0, 0]);

    /// Maximum value (all bits set).
    pub const MAX: Self = Self::from_limbs([u64::MAX; 4]);

    /// Construct from four `u64` limbs in little-endian word order
    /// (`limbs[0]` = least significant).  This is a `const fn`.
    #[inline]
    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self(limbs)
    }

    /// Reference to the internal limbs in little-endian word order.
    #[inline]
    pub const fn as_limbs(&self) -> &[u64; 4] {
        &self.0
    }

    /// Mutable reference to the internal limbs.
    ///
    /// # Safety
    /// The caller must not violate the invariants of the underlying integer
    /// representation (all 256 bits are significant; no unused-bit invariants).
    #[inline]
    pub unsafe fn as_limbs_mut(&mut self) -> &mut [u64; 4] {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// Byte conversion — const fn paths use ruint directly.
// ---------------------------------------------------------------------------

impl U256 {
    /// Construct from a big-endian fixed-size byte array.
    /// Panics if `BYTES != 32`.
    #[inline]
    pub const fn from_be_bytes<const BYTES: usize>(bytes: [u8; BYTES]) -> Self {
        Self(*alloy_primitives::U256::from_be_bytes::<BYTES>(bytes).as_limbs())
    }

    /// Construct from a big-endian byte slice.
    /// Panics if `bytes.len() != 32`.
    #[inline]
    pub const fn from_be_slice(bytes: &[u8]) -> Self {
        Self(*alloy_primitives::U256::from_be_slice(bytes).as_limbs())
    }

    /// Construct from a big-endian byte slice, returning `None` if the length
    /// is not exactly 32.
    #[inline]
    pub const fn try_from_be_slice(bytes: &[u8]) -> Option<Self> {
        match alloy_primitives::U256::try_from_be_slice(bytes) {
            Some(v) => Some(Self(*v.as_limbs())),
            None => None,
        }
    }

    /// Convert to a big-endian byte array of exactly `BYTES` bytes.
    /// Panics if `BYTES != 32`.
    #[inline]
    pub const fn to_be_bytes<const BYTES: usize>(&self) -> [u8; BYTES] {
        ru(self.0).to_be_bytes::<BYTES>()
    }

    /// Convert to a big-endian byte vector (32 bytes).
    #[cfg(feature = "std")]
    #[inline]
    pub fn to_be_bytes_vec(&self) -> std::vec::Vec<u8> {
        self.to_be_bytes::<32>().to_vec()
    }

    /// Convert to a big-endian byte vector with leading zero bytes stripped.
    #[cfg(feature = "std")]
    #[inline]
    pub fn to_be_bytes_trimmed_vec(&self) -> std::vec::Vec<u8> {
        let bytes = self.to_be_bytes::<32>();
        let start = bytes.iter().position(|&b| b != 0).unwrap_or(32);
        bytes[start..].to_vec()
    }
}

// ---------------------------------------------------------------------------
// Const-fn arithmetic — always use ruint directly.
// Used for compile-time constants; cannot go through a trait object.
// ---------------------------------------------------------------------------

impl U256 {
    /// Wrapping (modular) addition. `const fn`.
    #[inline]
    pub const fn wrapping_add(self, rhs: Self) -> Self {
        Self(*ru(self.0).wrapping_add(ru(rhs.0)).as_limbs())
    }

    /// Wrapping (modular) subtraction. `const fn`.
    #[inline]
    pub const fn wrapping_sub(self, rhs: Self) -> Self {
        Self(*ru(self.0).wrapping_sub(ru(rhs.0)).as_limbs())
    }

    /// Saturating addition. `const fn`.
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(*ru(self.0).saturating_add(ru(rhs.0)).as_limbs())
    }

    /// Saturating subtraction. `const fn`.
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(*ru(self.0).saturating_sub(ru(rhs.0)).as_limbs())
    }

    /// Checked addition. Returns `None` on overflow. `const fn`.
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        match ru(self.0).checked_add(ru(rhs.0)) {
            Some(v) => Some(Self(*v.as_limbs())),
            None => None,
        }
    }

    /// Checked subtraction. Returns `None` on underflow. `const fn`.
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        match ru(self.0).checked_sub(ru(rhs.0)) {
            Some(v) => Some(Self(*v.as_limbs())),
            None => None,
        }
    }

    /// Overflowing addition. `const fn`.
    #[inline]
    pub const fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let (v, o) = ru(self.0).overflowing_add(ru(rhs.0));
        (Self(*v.as_limbs()), o)
    }

    /// Overflowing subtraction. `const fn`.
    #[inline]
    pub const fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let (v, o) = ru(self.0).overflowing_sub(ru(rhs.0));
        (Self(*v.as_limbs()), o)
    }
}

// ---------------------------------------------------------------------------
// Runtime arithmetic — delegate to the injectable backend.
// ---------------------------------------------------------------------------

impl U256 {
    /// Wrapping multiplication. Bypasses backend dispatch — uses ruint directly.
    #[inline]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        Self(*ru(self.0).wrapping_mul(ru(rhs.0)).as_limbs())
    }

    /// Wrapping division. Panics if `rhs` is zero.
    #[inline]
    pub fn wrapping_div(self, rhs: Self) -> Self {
        Self(ops().wrapping_div(self.0, rhs.0))
    }

    /// Wrapping remainder. Panics if `rhs` is zero.
    #[inline]
    pub fn wrapping_rem(self, rhs: Self) -> Self {
        Self(ops().wrapping_rem(self.0, rhs.0))
    }

    /// Wrapping (two's-complement) negation. Bypasses backend dispatch — uses ruint directly.
    #[inline]
    pub fn wrapping_neg(self) -> Self {
        Self(*ru(self.0).wrapping_neg().as_limbs())
    }

    /// Saturating multiplication.
    #[inline]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        Self(ops().saturating_mul(self.0, rhs.0))
    }

    /// Checked multiplication. Returns `None` on overflow.
    #[inline]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        ops().checked_mul(self.0, rhs.0).map(Self)
    }

    /// Checked division. Returns `None` if `rhs` is zero.
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        ops().checked_div(self.0, rhs.0).map(Self)
    }

    /// Overflowing multiplication.
    #[inline]
    pub fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        let (v, o) = ops().overflowing_mul(self.0, rhs.0);
        (Self(v), o)
    }

    /// Exponentiation: `self ^ exp`.
    #[inline]
    pub fn pow(self, exp: Self) -> Self {
        Self(ops().pow(self.0, exp.0))
    }

    /// Modular addition: `(self + rhs) % modulus`. Bypasses backend dispatch — uses ruint directly.
    #[inline]
    pub fn add_mod(self, rhs: Self, modulus: Self) -> Self {
        Self(*ru(self.0).add_mod(ru(rhs.0), ru(modulus.0)).as_limbs())
    }

    /// Modular multiplication: `(self * rhs) % modulus`. Bypasses backend dispatch — uses ruint directly.
    #[inline]
    pub fn mul_mod(self, rhs: Self, modulus: Self) -> Self {
        Self(*ru(self.0).mul_mod(ru(rhs.0), ru(modulus.0)).as_limbs())
    }

    /// Arithmetic (signed) right shift. `shift` must be < 256. Bypasses backend dispatch — uses ruint directly.
    #[inline]
    pub fn arithmetic_shr(self, shift: usize) -> Self {
        Self(*ru(self.0).arithmetic_shr(shift).as_limbs())
    }
}

// ---------------------------------------------------------------------------
// Generic numeric conversions (use ruint directly — not overridable).
// ---------------------------------------------------------------------------

impl U256 {
    /// Saturating cast to `T`. Returns `T::MAX` if `self` exceeds the range.
    #[inline]
    pub fn saturating_to<T>(&self) -> T
    where
        alloy_primitives::U256: ruint::UintTryTo<T>,
    {
        ru(self.0).saturating_to()
    }

    /// Lossless cast to `T`. Panics if the value does not fit.
    #[inline]
    pub fn to<T>(&self) -> T
    where
        alloy_primitives::U256: ruint::UintTryTo<T>,
        T: core::fmt::Debug,
    {
        ru(self.0).to()
    }
}

// ---------------------------------------------------------------------------
// Bitwise inspection
// ---------------------------------------------------------------------------

impl U256 {
    /// Returns `true` if the value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == [0; 4]
    }

    /// Const-compatible zero check.
    #[inline]
    pub const fn const_is_zero(&self) -> bool {
        self.0[0] == 0 && self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0
    }

    /// Const-compatible equality check.
    #[inline]
    pub const fn const_eq(&self, other: &Self) -> bool {
        self.0[0] == other.0[0]
            && self.0[1] == other.0[1]
            && self.0[2] == other.0[2]
            && self.0[3] == other.0[3]
    }

    /// Returns the bit at `index` (0 = least significant).
    #[inline]
    pub const fn bit(&self, index: usize) -> bool {
        ru(self.0).bit(index)
    }

    /// Returns the byte at `index` (0 = least significant byte, little-endian).
    #[inline]
    pub const fn byte(&self, index: usize) -> u8 {
        ru(self.0).byte(index)
    }

    /// Number of leading zero bits.
    #[inline]
    pub const fn leading_zeros(&self) -> usize {
        ru(self.0).leading_zeros()
    }

    /// Number of bits needed to represent this value (bit length). Returns 0 for zero.
    #[inline]
    pub const fn bit_len(&self) -> usize {
        ru(self.0).bit_len()
    }
}

// ---------------------------------------------------------------------------
// Operator implementations — delegate to backend.
// ---------------------------------------------------------------------------

// Arithmetic: wrapping semantics (overflow is discarded).
// Add/Sub bypass backend dispatch — use ruint const-fn paths directly.
// Mul/Div/Rem delegate to the installed backend (overridable for acceleration).
impl core::ops::Add for U256 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}
impl core::ops::Sub for U256 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}
impl core::ops::Mul for U256 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(ops().overflowing_mul(self.0, rhs.0).0)
    }
}
impl core::ops::Div for U256 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self(ops().div(self.0, rhs.0))
    }
}
impl core::ops::Rem for U256 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Self(ops().rem(self.0, rhs.0))
    }
}

// Bitwise — bypass backend dispatch, operate directly on limbs / ruint.
impl core::ops::Not for U256 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }
}
impl core::ops::BitAnd for U256 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self([
            self.0[0] & rhs.0[0],
            self.0[1] & rhs.0[1],
            self.0[2] & rhs.0[2],
            self.0[3] & rhs.0[3],
        ])
    }
}
impl core::ops::BitOr for U256 {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self([
            self.0[0] | rhs.0[0],
            self.0[1] | rhs.0[1],
            self.0[2] | rhs.0[2],
            self.0[3] | rhs.0[3],
        ])
    }
}
impl core::ops::BitXor for U256 {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self([
            self.0[0] ^ rhs.0[0],
            self.0[1] ^ rhs.0[1],
            self.0[2] ^ rhs.0[2],
            self.0[3] ^ rhs.0[3],
        ])
    }
}

// Shifts — bypass backend dispatch, use ruint directly.
impl core::ops::Shl<usize> for U256 {
    type Output = Self;
    #[inline]
    fn shl(self, rhs: usize) -> Self {
        Self(*(ru(self.0) << rhs).as_limbs())
    }
}
impl core::ops::Shr<usize> for U256 {
    type Output = Self;
    #[inline]
    fn shr(self, rhs: usize) -> Self {
        Self(*(ru(self.0) >> rhs).as_limbs())
    }
}
impl core::ops::ShlAssign<usize> for U256 {
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        *self = *self << rhs;
    }
}
impl core::ops::ShrAssign<usize> for U256 {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        *self = *self >> rhs;
    }
}

// Assign variants.
macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $op_trait:ident, $op_method:ident) => {
        impl core::ops::$trait for U256 {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                *self = core::ops::$op_trait::$op_method(*self, rhs);
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, Add, add);
impl_assign_op!(SubAssign, sub_assign, Sub, sub);
impl_assign_op!(MulAssign, mul_assign, Mul, mul);
impl_assign_op!(DivAssign, div_assign, Div, div);
impl_assign_op!(RemAssign, rem_assign, Rem, rem);
impl_assign_op!(BitAndAssign, bitand_assign, BitAnd, bitand);
impl_assign_op!(BitOrAssign, bitor_assign, BitOr, bitor);
impl_assign_op!(BitXorAssign, bitxor_assign, BitXor, bitxor);

// ---------------------------------------------------------------------------
// From impls for primitive types
// ---------------------------------------------------------------------------

macro_rules! impl_from_primitive {
    ($($t:ty),+) => {
        $(
            impl From<$t> for U256 {
                #[inline]
                fn from(v: $t) -> Self {
                    Self(*alloy_primitives::U256::from(v).as_limbs())
                }
            }
        )+
    };
}

impl_from_primitive!(u8, u16, u32, u64, u128, usize, bool);

// Signed integer conversions: two's-complement sign extension (EVM semantics).
macro_rules! impl_from_signed {
    ($($t:ty as $ut:ty),+) => {
        $(
            impl From<$t> for U256 {
                #[inline]
                fn from(v: $t) -> Self {
                    if v >= 0 {
                        Self::from(v as $ut)
                    } else {
                        Self::MAX - Self::from(!(v as $ut))
                    }
                }
            }
        )+
    };
}

impl_from_signed!(i8 as u8, i16 as u16, i32 as u32, i64 as u64);

// ---------------------------------------------------------------------------
// TryFrom impls (narrowing conversions)
// ---------------------------------------------------------------------------

impl TryFrom<U256> for u64 {
    type Error = &'static str;
    #[inline]
    fn try_from(v: U256) -> Result<Self, Self::Error> {
        if v.0[1] != 0 || v.0[2] != 0 || v.0[3] != 0 {
            Err("U256 value too large for u64")
        } else {
            Ok(v.0[0])
        }
    }
}

impl TryFrom<U256> for u128 {
    type Error = &'static str;
    #[inline]
    fn try_from(v: U256) -> Result<Self, Self::Error> {
        if v.0[2] != 0 || v.0[3] != 0 {
            Err("U256 value too large for u128")
        } else {
            Ok(v.0[0] as u128 | ((v.0[1] as u128) << 64))
        }
    }
}

impl TryFrom<U256> for usize {
    type Error = &'static str;
    #[inline]
    fn try_from(v: U256) -> Result<Self, Self::Error> {
        let lo: u64 = v.try_into()?;
        usize::try_from(lo).map_err(|_| "U256 value too large for usize")
    }
}

// ---------------------------------------------------------------------------
// Interop with alloy_primitives::U256
// ---------------------------------------------------------------------------

impl From<alloy_primitives::U256> for U256 {
    #[inline]
    fn from(v: alloy_primitives::U256) -> Self {
        Self(*v.as_limbs())
    }
}

impl From<U256> for alloy_primitives::U256 {
    #[inline]
    fn from(v: U256) -> Self {
        alloy_primitives::U256::from_limbs(v.0)
    }
}

// FixedBytes<32> / B256 interop (bytes are big-endian).
impl From<alloy_primitives::FixedBytes<32>> for U256 {
    #[inline]
    fn from(v: alloy_primitives::FixedBytes<32>) -> Self {
        Self::from_be_bytes(v.0)
    }
}

impl From<U256> for alloy_primitives::FixedBytes<32> {
    #[inline]
    fn from(v: U256) -> Self {
        alloy_primitives::FixedBytes(v.to_be_bytes::<32>())
    }
}

// ---------------------------------------------------------------------------
// FromStr
// ---------------------------------------------------------------------------

impl core::str::FromStr for U256 {
    type Err = <alloy_primitives::U256 as core::str::FromStr>::Err;
    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        alloy_primitives::U256::from_str(s).map(|v| Self(*v.as_limbs()))
    }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

impl core::fmt::Debug for U256 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&ru(self.0), f)
    }
}

impl core::fmt::Display for U256 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&ru(self.0), f)
    }
}

impl core::fmt::LowerHex for U256 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::LowerHex::fmt(&ru(self.0), f)
    }
}

impl core::fmt::UpperHex for U256 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::UpperHex::fmt(&ru(self.0), f)
    }
}
