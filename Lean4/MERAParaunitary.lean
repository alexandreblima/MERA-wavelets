/-
  Formalization of Theorem 1: MERA-Paraunitary Architectural Equivalence
  Paper: "Adaptive Wavelets for Backbone Telemetry in 6G Digital Twins"
  Authors: Lima, Hesselbach, Amazonas

  This file formalizes the equivalence between:
  - MERA-inspired layers with orthogonal matrices U ∈ O(2)
  - Two-channel paraunitary filter banks with constant polyphase matrix
-/

import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.Orthogonal
import Mathlib.LinearAlgebra.UnitaryGroup
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic

open Matrix
open scoped Matrix

/-!
## Preliminary Definitions

### Definition 2 (Orthogonal Transformation)
For real spaces H = ℝⁿ, a matrix U ∈ ℝⁿˣⁿ is orthogonal if U†U = I,
where † denotes transpose for real matrices.
-/

/-- Orthogonal group O(n): matrices U such that Uᵀ * U = I -/
def OrthogonalGroup (n : ℕ) : Set (Matrix (Fin n) (Fin n) ℝ) :=
  { U | Uᵀ * U = 1 }

/-- O(2) notation for the specific case in the paper -/
abbrev O2 := OrthogonalGroup 2

/-!
### Definition 3 (MERA-Inspired Orthogonal Layer)

Let x ∈ ℝᴺ be a discrete signal with N = 2^L samples.
A MERA layer at scale ℓ applies a 2×2 orthogonal matrix Uℓ ∈ O(2)
to disjoint pairs of samples, followed by implicit downsampling ↓2:

  ⎡ a_k^(ℓ) ⎤     ⎡ x_{2k}   ⎤
  ⎣ d_k^(ℓ) ⎦ = Uℓ ⎣ x_{2k+1} ⎦,   k = 0, ..., N/2^ℓ - 1
-/

/-- MERA layer: applies orthogonal matrix U to pair of adjacent samples -/
structure MERALayer where
  U : Matrix (Fin 2) (Fin 2) ℝ
  h_ortho : Uᵀ * U = 1

/-- Approximation and detail coefficients produced by a MERA layer -/
structure MERAOutput where
  a : ℝ  -- approximation coefficient (lowpass)
  d : ℝ  -- detail coefficient (highpass)

/-- Application of a MERA layer to a pair of samples -/
def MERALayer.apply (layer : MERALayer) (x0 x1 : ℝ) : MERAOutput :=
  let input : Fin 2 → ℝ := ![x0, x1]
  let output := layer.U.mulVec input
  ⟨output 0, output 1⟩

/-!
### Definition 4 (Paraunitary Filter Bank)

A two-channel filter bank with polyphase matrix

  E(z) = ⎡ G₀(z)  G₁(z) ⎤
         ⎣ H₀(z)  H₁(z) ⎦

is paraunitary if E(z) E†(z⁻¹) = I.

For a CONSTANT polyphase matrix E(z) ≡ U (independent of z),
the paraunitarity condition reduces to:
  U * U† = I  (orthogonality)
-/

/-- Constant polyphase matrix of a 2-channel filter bank -/
structure PolyphaseMatrix where
  g0 : ℝ  -- even component of lowpass filter
  g1 : ℝ  -- odd component of lowpass filter
  h0 : ℝ  -- even component of highpass filter
  h1 : ℝ  -- odd component of highpass filter

/-- Converts polyphase matrix to Matrix (Fin 2) (Fin 2) ℝ -/
def PolyphaseMatrix.toMatrix (E : PolyphaseMatrix) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![E.g0, E.g1; E.h0, E.h1]

/-- Paraunitarity condition for constant matrix: Uᵀ * U = I -/
def PolyphaseMatrix.isParaunitary (E : PolyphaseMatrix) : Prop :=
  E.toMatrixᵀ * E.toMatrix = 1

/-!
### 2-tap FIR Filters

By the Noble Identities (Eq. 10-11 in the paper), when E(z) ≡ U is constant,
the analysis filters are 2-tap FIR:

  G(z) = g₀ + g₁z⁻¹
  H(z) = h₀ + h₁z⁻¹
-/

/-- 2-tap FIR filter -/
structure TwoTapFilter where
  coef0 : ℝ  -- coefficient at z⁰
  coef1 : ℝ  -- coefficient at z⁻¹

/-- Extracts lowpass filter from polyphase matrix -/
def PolyphaseMatrix.lowpassFilter (E : PolyphaseMatrix) : TwoTapFilter :=
  ⟨E.g0, E.g1⟩

/-- Extracts highpass filter from polyphase matrix -/
def PolyphaseMatrix.highpassFilter (E : PolyphaseMatrix) : TwoTapFilter :=
  ⟨E.h0, E.h1⟩

/-!
## Theorem 1: Architectural Equivalence

A MERA-inspired layer (Definition 3) is equivalent to a two-channel
paraunitary filter bank whose polyphase representation is a constant
orthonormal matrix E(z) ≡ Uℓ.

The proof establishes a bijection between:
- {MERA layers with U ∈ O(2)}
- {Paraunitary filter banks with constant polyphase matrix}
-/

/-!
### Direction 1: Sufficiency (MERA → Paraunitary Filter Bank)

Given: Local MERA operator Uℓ ∈ O(2) acting on adjacent pairs
Prove: This induces a paraunitary filter bank with E(z) ≡ Uℓ
-/

/-- Constructs polyphase matrix from MERA layer -/
def MERALayer.toPolyphase (layer : MERALayer) : PolyphaseMatrix where
  g0 := layer.U 0 0
  g1 := layer.U 0 1
  h0 := layer.U 1 0
  h1 := layer.U 1 1

/-- Lemma: The polyphase matrix derived from MERA coincides with U -/
lemma MERALayer.toPolyphase_toMatrix_eq (layer : MERALayer) :
    layer.toPolyphase.toMatrix = layer.U := by
  ext i j
  fin_cases i <;> fin_cases j <;> rfl

/-- Sufficiency: MERA layer induces paraunitary filter bank -/
theorem mera_implies_paraunitary (layer : MERALayer) :
    layer.toPolyphase.isParaunitary := by
  unfold PolyphaseMatrix.isParaunitary
  rw [layer.toPolyphase_toMatrix_eq]
  exact layer.h_ortho

/-!
### Direction 2: Necessity (Paraunitary Filter Bank → MERA)

Given: Paraunitary filter bank with constant polyphase matrix E(z) ≡ U ∈ O(2)
Prove: The analysis operation is precisely a MERA layer
-/

/-- Constructs MERA layer from paraunitary polyphase matrix -/
def PolyphaseMatrix.toMERALayer (E : PolyphaseMatrix) (h : E.isParaunitary) : MERALayer where
  U := E.toMatrix
  h_ortho := h

/-- Necessity: Constant paraunitary filter bank implements MERA -/
theorem paraunitary_implies_mera (E : PolyphaseMatrix) (h : E.isParaunitary) :
    ∃ (layer : MERALayer), layer.U = E.toMatrix :=
  ⟨E.toMERALayer h, rfl⟩

/-!
### Theorem 1 (Complete Statement): Architectural Equivalence

The equivalence is established by the bijection:
- MERA → Polyphase: MERALayer.toPolyphase
- Polyphase → MERA: PolyphaseMatrix.toMERALayer
-/

/-- Theorem 1: Equivalence MERA ↔ Paraunitary Filter Bank -/
theorem architectural_equivalence :
    (∀ layer : MERALayer, layer.toPolyphase.isParaunitary) ∧
    (∀ E : PolyphaseMatrix, E.isParaunitary → ∃ layer : MERALayer, layer.U = E.toMatrix) := by
  constructor
  · intro layer
    exact mera_implies_paraunitary layer
  · intro E hE
    exact paraunitary_implies_mera E hE

/-!
## Proof via Z-Transform (Frequency Domain)

This section formalizes the proof of Theorem 1 through the Z-domain,
following the steps in the paper (Eq. 10-31).

### Z-Domain Representation

For 2-tap FIR filters, the Z-transform is:
  G(z) = g₀ + g₁z⁻¹
  H(z) = h₀ + h₁z⁻¹

The variable z represents the delay operator: z⁻¹ · x[n] = x[n-1]
-/

/-- Representation of an FIR filter in the Z-domain as a function ℂ → ℂ -/
noncomputable def TwoTapFilter.zTransform (f : TwoTapFilter) (z : ℂ) : ℂ :=
  f.coef0 + f.coef1 * z⁻¹

/-- Frequency response: evaluation at z = e^{jω} -/
noncomputable def TwoTapFilter.freqResponse (f : TwoTapFilter) (ω : ℝ) : ℂ :=
  f.zTransform (Complex.exp (Complex.I * ω))

/-!
### Polyphase Decomposition (Eq. 7-9 in the paper)

The type-1 polyphase decomposition separates even and odd coefficients:
  G(z) = G₀(z²) + z⁻¹G₁(z²)

For 2-tap filters:
  G₀(z) = g₀  (constant)
  G₁(z) = g₁  (constant)

Therefore: G(z) = g₀ + g₁z⁻¹
-/

/-- Even polyphase component (for 2-tap filter, it's just g₀) -/
def TwoTapFilter.polyphaseEven (f : TwoTapFilter) : ℝ := f.coef0

/-- Odd polyphase component (for 2-tap filter, it's just g₁) -/
def TwoTapFilter.polyphaseOdd (f : TwoTapFilter) : ℝ := f.coef1

/-!
### Noble Identities (Eq. 10-11)

For 2-tap filters with constant polyphase components:
  G(z) = g₀ + g₁z⁻¹
  H(z) = h₀ + h₁z⁻¹

These are Eq. 22-23 and 30-31 in the paper.
-/

/-- Noble Identity for lowpass: G(z) = g₀ + g₁z⁻¹ -/
theorem noble_identity_lowpass (E : PolyphaseMatrix) (z : ℂ) :
    E.lowpassFilter.zTransform z = E.g0 + E.g1 * z⁻¹ := by
  rfl

/-- Noble Identity for highpass: H(z) = h₀ + h₁z⁻¹ -/
theorem noble_identity_highpass (E : PolyphaseMatrix) (z : ℂ) :
    E.highpassFilter.zTransform z = E.h0 + E.h1 * z⁻¹ := by
  rfl

/-!
### Paraunitarity in the Z-Domain

For constant polyphase matrix E, paraunitarity E(z)E†(z⁻¹) = I
reduces to the orthogonality condition E·Eᵀ = I.

This is because:
- E(z) = E (constant, independent of z)
- E†(z⁻¹) = Eᵀ (conjugate = transpose for real matrices)
-/

/-- Evaluation of polyphase matrix at z (constant, so independent of z) -/
noncomputable def PolyphaseMatrix.evalZ (E : PolyphaseMatrix) (_z : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![E.g0, E.g1; E.h0, E.h1]

/-- Paraconjugate for constant real matrix: E†(z⁻¹) = Eᵀ -/
noncomputable def PolyphaseMatrix.paraconjugate (E : PolyphaseMatrix) (_z : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![E.g0, E.h0; E.g1, E.h1]

/-- Paraunitarity product in Z-domain -/
noncomputable def PolyphaseMatrix.paraunitaryProduct (E : PolyphaseMatrix) (z : ℂ) :
    Matrix (Fin 2) (Fin 2) ℂ :=
  E.evalZ z * E.paraconjugate z

/-- Paraunitarity condition in Z-domain -/
def PolyphaseMatrix.isParaunitaryZ (E : PolyphaseMatrix) : Prop :=
  ∀ z : ℂ, E.paraunitaryProduct z = 1

/-- Auxiliary lemma: Uᵀ·U = I implies U·Uᵀ = I for 2×2 matrices -/
lemma orthogonal_transpose_comm {U : Matrix (Fin 2) (Fin 2) ℝ} (h : Uᵀ * U = 1) :
    U * Uᵀ = 1 := by
  -- Extract the 4 equations from h : Uᵀ * U = 1 (orthonormal columns)
  have c00 : U 0 0 * U 0 0 + U 1 0 * U 1 0 = 1 := by
    have := congr_fun (congr_fun h 0) 0
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply,
               Fin.sum_univ_two, Fin.reduceEq, ↓reduceIte] at this
    exact this
  have c01 : U 0 0 * U 0 1 + U 1 0 * U 1 1 = 0 := by
    have := congr_fun (congr_fun h 0) 1
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply,
               Fin.sum_univ_two, Fin.reduceEq, ↓reduceIte] at this
    exact this
  have c11 : U 0 1 * U 0 1 + U 1 1 * U 1 1 = 1 := by
    have := congr_fun (congr_fun h 1) 1
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply,
               Fin.sum_univ_two, Fin.reduceEq, ↓reduceIte] at this
    exact this
  -- det(U)² = 1 (from det(UᵀU) = det(I) = 1)
  have hdet : (U 0 0 * U 1 1 - U 0 1 * U 1 0) ^ 2 = 1 := by
    have hdetUU : (Uᵀ * U).det = (1 : Matrix (Fin 2) (Fin 2) ℝ).det := by rw [h]
    simp only [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at hdetUU
    have hUsq : U.det ^ 2 = 1 := by nlinarith
    simp only [Matrix.det_fin_two] at hUsq
    linarith
  -- Prove U * Uᵀ = 1 (orthonormal rows) - entry by entry
  have r00 : U 0 0 * U 0 0 + U 0 1 * U 0 1 = 1 := by
    nlinarith [sq_nonneg (U 0 0 * U 1 0 + U 0 1 * U 1 1), sq_nonneg (U 0 0 * U 1 1 - U 0 1 * U 1 0),
               sq_nonneg (U 0 0), sq_nonneg (U 0 1), sq_nonneg (U 1 0), sq_nonneg (U 1 1)]
  have r01 : U 0 0 * U 1 0 + U 0 1 * U 1 1 = 0 := by nlinarith [sq_nonneg (U 0 0 * U 0 1 + U 1 0 * U 1 1)]
  have r10 : U 1 0 * U 0 0 + U 1 1 * U 0 1 = 0 := by nlinarith [sq_nonneg (U 0 0 * U 0 1 + U 1 0 * U 1 1)]
  have r11 : U 1 0 * U 1 0 + U 1 1 * U 1 1 = 1 := by
    nlinarith [sq_nonneg (U 0 0 * U 1 0 + U 0 1 * U 1 1), sq_nonneg (U 0 0 * U 1 1 - U 0 1 * U 1 0),
               sq_nonneg (U 0 0), sq_nonneg (U 0 1), sq_nonneg (U 1 0), sq_nonneg (U 1 1)]
  -- Construct the result matrix
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply,
               Fin.sum_univ_two, Fin.isValue, Fin.reduceEq, ↓reduceIte]
  · exact r00
  · exact r01
  · exact r10
  · exact r11

/-- Theorem 1 (via Z-domain): Orthogonality ⟺ Z-domain Paraunitarity -/
theorem theorem1_z_domain (E : PolyphaseMatrix) :
    E.isParaunitary ↔ E.isParaunitaryZ := by
  constructor
  · -- (→) Orthogonal implies paraunitary in Z-domain
    intro h_ortho
    unfold PolyphaseMatrix.isParaunitaryZ PolyphaseMatrix.paraunitaryProduct
    intro z
    unfold PolyphaseMatrix.isParaunitary at h_ortho
    have h_EEt : E.toMatrix * E.toMatrixᵀ = 1 := orthogonal_transpose_comm h_ortho
    -- Extract the 4 equations from h_EEt
    have r00 : E.g0 * E.g0 + E.g1 * E.g1 = 1 := by
      have := congr_fun (congr_fun h_EEt 0) 0
      simp only [PolyphaseMatrix.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
                 Matrix.one_apply, Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val',
                 Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
                 Fin.reduceEq, ↓reduceIte] at this
      exact this
    have r01 : E.g0 * E.h0 + E.g1 * E.h1 = 0 := by
      have := congr_fun (congr_fun h_EEt 0) 1
      simp only [PolyphaseMatrix.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
                 Matrix.one_apply, Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val',
                 Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
                 Fin.reduceEq, ↓reduceIte] at this
      exact this
    have r10 : E.h0 * E.g0 + E.h1 * E.g1 = 0 := by
      have := congr_fun (congr_fun h_EEt 1) 0
      simp only [PolyphaseMatrix.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
                 Matrix.one_apply, Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val',
                 Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
                 Fin.reduceEq, ↓reduceIte] at this
      exact this
    have r11 : E.h0 * E.h0 + E.h1 * E.h1 = 1 := by
      have := congr_fun (congr_fun h_EEt 1) 1
      simp only [PolyphaseMatrix.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
                 Matrix.one_apply, Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val',
                 Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
                 Fin.reduceEq, ↓reduceIte] at this
      exact this
    -- Prove entry by entry via direct calculation
    have eq00 : (E.evalZ z * E.paraconjugate z) 0 0 = 1 := by
      simp only [PolyphaseMatrix.evalZ, PolyphaseMatrix.paraconjugate, Matrix.mul_apply,
                 Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
                 Matrix.head_cons]
      simp only [← Complex.ofReal_mul, ← Complex.ofReal_add, ← Complex.ofReal_one]
      exact congrArg Complex.ofReal r00
    have eq01 : (E.evalZ z * E.paraconjugate z) 0 1 = 0 := by
      simp only [PolyphaseMatrix.evalZ, PolyphaseMatrix.paraconjugate, Matrix.mul_apply,
                 Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
                 Matrix.head_cons]
      simp only [← Complex.ofReal_mul, ← Complex.ofReal_add, ← Complex.ofReal_zero]
      exact congrArg Complex.ofReal r01
    have eq10 : (E.evalZ z * E.paraconjugate z) 1 0 = 0 := by
      simp only [PolyphaseMatrix.evalZ, PolyphaseMatrix.paraconjugate, Matrix.mul_apply,
                 Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
                 Matrix.head_cons]
      simp only [← Complex.ofReal_mul, ← Complex.ofReal_add, ← Complex.ofReal_zero]
      exact congrArg Complex.ofReal r10
    have eq11 : (E.evalZ z * E.paraconjugate z) 1 1 = 1 := by
      simp only [PolyphaseMatrix.evalZ, PolyphaseMatrix.paraconjugate, Matrix.mul_apply,
                 Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
                 Matrix.head_cons]
      simp only [← Complex.ofReal_mul, ← Complex.ofReal_add, ← Complex.ofReal_one]
      exact congrArg Complex.ofReal r11
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp only [Matrix.one_apply, Fin.isValue, Fin.reduceEq, ↓reduceIte]
    · exact eq00
    · exact eq01
    · exact eq10
    · exact eq11
  · -- (←) Paraunitary in Z-domain implies orthogonal
    intro h_paraZ
    unfold PolyphaseMatrix.isParaunitary
    unfold PolyphaseMatrix.isParaunitaryZ PolyphaseMatrix.paraunitaryProduct at h_paraZ
    specialize h_paraZ 1
    -- Extract the 4 equations from h_paraZ and convert ℂ → ℝ
    have r00 : E.g0 * E.g0 + E.g1 * E.g1 = 1 := by
      have := congr_fun (congr_fun h_paraZ 0) 0
      simp only [PolyphaseMatrix.evalZ, PolyphaseMatrix.paraconjugate, Matrix.mul_apply,
                 Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
                 Matrix.head_cons, Matrix.one_apply_eq] at this
      simp only [← Complex.ofReal_mul, ← Complex.ofReal_add, ← Complex.ofReal_one,
                 Complex.ofReal_inj] at this
      exact this
    have r01 : E.g0 * E.h0 + E.g1 * E.h1 = 0 := by
      have := congr_fun (congr_fun h_paraZ 0) 1
      simp only [PolyphaseMatrix.evalZ, PolyphaseMatrix.paraconjugate, Matrix.mul_apply,
                 Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
                 Matrix.head_cons, Matrix.one_apply, Fin.reduceEq, ↓reduceIte] at this
      simp only [← Complex.ofReal_mul, ← Complex.ofReal_add, ← Complex.ofReal_zero,
                 Complex.ofReal_inj] at this
      exact this
    have r10 : E.h0 * E.g0 + E.h1 * E.g1 = 0 := by
      have := congr_fun (congr_fun h_paraZ 1) 0
      simp only [PolyphaseMatrix.evalZ, PolyphaseMatrix.paraconjugate, Matrix.mul_apply,
                 Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
                 Matrix.head_cons, Matrix.one_apply, Fin.reduceEq, ↓reduceIte] at this
      simp only [← Complex.ofReal_mul, ← Complex.ofReal_add, ← Complex.ofReal_zero,
                 Complex.ofReal_inj] at this
      exact this
    have r11 : E.h0 * E.h0 + E.h1 * E.h1 = 1 := by
      have := congr_fun (congr_fun h_paraZ 1) 1
      simp only [PolyphaseMatrix.evalZ, PolyphaseMatrix.paraconjugate, Matrix.mul_apply,
                 Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
                 Matrix.head_cons, Matrix.one_apply_eq] at this
      simp only [← Complex.ofReal_mul, ← Complex.ofReal_add, ← Complex.ofReal_one,
                 Complex.ofReal_inj] at this
      exact this
    -- Construct E.toMatrixᵀ * E.toMatrix = 1
    have hEEt : E.toMatrix * E.toMatrixᵀ = 1 := by
      ext i j
      simp only [PolyphaseMatrix.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
                 Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
                 Matrix.head_cons, Matrix.one_apply]
      fin_cases i <;> fin_cases j <;> simp [Fin.zero_eq_one_iff, Fin.reduceEq]
      · exact r00
      · exact r01
      · exact r10
      · exact r11
    -- Use lemma to get EᵀE = 1 from EEᵀ = 1
    exact orthogonal_transpose_comm hEEt

/-!
## Frequency Domain Analysis: Power Complementarity

For a paraunitary filter bank, the sum of magnitude-squared responses equals 2:
  |G(e^{jω})|² + |H(e^{jω})|² = 2

This is Parseval's identity for the 2-tap case.
Note: If the filters were normalized by 1/√2, we would have = 1.
-/

/-- Magnitude squared of frequency response -/
noncomputable def TwoTapFilter.magSquared (f : TwoTapFilter) (ω : ℝ) : ℝ :=
  Complex.normSq (f.freqResponse ω)

/-- Lemma: |g₀ + g₁ e^{-jω}|² = g₀² + g₁² + 2g₀g₁cos(ω)
    This is a standard trigonometric identity:
    |a + b e^{-jω}|² = a² + b² + 2ab cos(ω) -/
lemma two_tap_mag_squared (g0 g1 ω : ℝ) :
    Complex.normSq (g0 + g1 * (Complex.exp (Complex.I * ω))⁻¹) =
    g0^2 + g1^2 + 2 * g0 * g1 * Real.cos ω := by
  -- e^{-jω} = cos(-ω) + j sin(-ω) = cos(ω) - j sin(ω)
  have h_exp_inv : (Complex.exp (Complex.I * ω))⁻¹ = Complex.cos ω - Complex.I * Complex.sin ω := by
    rw [← Complex.exp_neg]
    conv_lhs => rw [show -(Complex.I * ↑ω) = ↑(-ω) * Complex.I by push_cast; ring]
    rw [Complex.exp_mul_I]
    simp only [Complex.ofReal_neg, Complex.cos_neg, Complex.sin_neg]
    ring
  rw [h_exp_inv]
  -- Expand |g₀ + g₁(cos ω - j sin ω)|² directly
  -- z = g₀ + g₁ cos ω - j g₁ sin ω, |z|² = Re(z)² + Im(z)²
  set z := (g0 : ℂ) + g1 * (Complex.cos ω - Complex.I * Complex.sin ω) with hz_def
  have hz_re : z.re = g0 + g1 * Real.cos ω := by
    simp only [hz_def, Complex.add_re, Complex.ofReal_re, Complex.mul_re, Complex.sub_re,
               Complex.cos_ofReal_re, Complex.I_re, Complex.sin_ofReal_re,
               Complex.I_im, Complex.sin_ofReal_im, Complex.cos_ofReal_im,
               Complex.ofReal_im, mul_zero, sub_zero, mul_one, zero_mul, add_zero]
  have hz_im : z.im = -g1 * Real.sin ω := by
    simp only [hz_def, Complex.add_im, Complex.ofReal_im, Complex.mul_im, Complex.sub_im,
               Complex.cos_ofReal_im, Complex.cos_ofReal_re, Complex.I_re,
               Complex.sin_ofReal_re, Complex.I_im, Complex.sin_ofReal_im,
               Complex.ofReal_re, mul_zero, add_zero, one_mul, zero_mul, sub_zero]
    ring
  rw [Complex.normSq_apply, hz_re, hz_im]
  have h_trig : g1^2 * Real.cos ω^2 + g1^2 * Real.sin ω^2 = g1^2 := by
    rw [← mul_add, Real.cos_sq_add_sin_sq, mul_one]
  ring_nf
  linarith [h_trig]

/-- Power complementarity: |G(ω)|² + |H(ω)|² = 2 for paraunitary filter bank -/
theorem power_complementarity (E : PolyphaseMatrix) (h : E.isParaunitary) (ω : ℝ) :
    E.lowpassFilter.magSquared ω + E.highpassFilter.magSquared ω = 2 := by
  unfold TwoTapFilter.magSquared TwoTapFilter.freqResponse TwoTapFilter.zTransform
  unfold PolyphaseMatrix.lowpassFilter PolyphaseMatrix.highpassFilter
  -- Apply the magnitude squared lemma
  rw [two_tap_mag_squared E.g0 E.g1 ω, two_tap_mag_squared E.h0 E.h1 ω]
  -- Extract orthogonality conditions
  have h_EEt : E.toMatrix * E.toMatrixᵀ = 1 := orthogonal_transpose_comm h
  -- Row 0 has norm 1: g0² + g1² = 1
  have row0_norm : E.g0^2 + E.g1^2 = 1 := by
    have := congr_fun (congr_fun h_EEt 0) 0
    simp only [PolyphaseMatrix.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
               Matrix.one_apply_eq, Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val',
               Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
               Fin.reduceEq, ↓reduceIte] at this
    have hmul : E.g0 * E.g0 = E.g0^2 := by ring
    have hmul' : E.g1 * E.g1 = E.g1^2 := by ring
    linarith
  -- Row 1 has norm 1: h0² + h1² = 1
  have row1_norm : E.h0^2 + E.h1^2 = 1 := by
    have := congr_fun (congr_fun h_EEt 1) 1
    simp only [PolyphaseMatrix.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
               Matrix.one_apply_eq, Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val',
               Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
               Fin.reduceEq, ↓reduceIte] at this
    have hmul : E.h0 * E.h0 = E.h0^2 := by ring
    have hmul' : E.h1 * E.h1 = E.h1^2 := by ring
    linarith
  -- Columns are orthogonal: g0*g1 + h0*h1 = 0
  have cols_cross : E.g0 * E.g1 + E.h0 * E.h1 = 0 := by
    have := congr_fun (congr_fun h 0) 1
    simp only [PolyphaseMatrix.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
               Matrix.one_apply, Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val',
               Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
               Fin.reduceEq, ↓reduceIte] at this
    linarith
  -- Combine
  calc E.g0^2 + E.g1^2 + 2 * E.g0 * E.g1 * Real.cos ω +
       (E.h0^2 + E.h1^2 + 2 * E.h0 * E.h1 * Real.cos ω)
      = (E.g0^2 + E.g1^2) + (E.h0^2 + E.h1^2) +
        2 * (E.g0 * E.g1 + E.h0 * E.h1) * Real.cos ω := by ring
    _ = 1 + 1 + 2 * 0 * Real.cos ω := by rw [row0_norm, row1_norm, cols_cross]
    _ = 2 := by ring

/-!
### Energy Conservation (Parseval's Identity)

For U ∈ O(2), orthogonality U†U = I guarantees:
  ‖x‖² = ‖a‖² + ‖d‖²

where (a, d) are the approximation and detail coefficients.
-/

/-- Local energy conservation: ‖output‖² = ‖input‖² -/
theorem energy_conservation (layer : MERALayer) (x0 x1 : ℝ) :
    let out := layer.apply x0 x1
    out.a ^ 2 + out.d ^ 2 = x0 ^ 2 + x1 ^ 2 := by
  -- Extract orthogonality conditions from the hypothesis layer.h_ortho
  have h := layer.h_ortho
  -- Matrix equality implies entry-wise equality
  have h00 : layer.U 0 0 ^ 2 + layer.U 1 0 ^ 2 = 1 := by
    have := congr_fun (congr_fun h 0) 0
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply_eq,
               Fin.sum_univ_two] at this
    linarith [sq_abs (layer.U 0 0), sq_abs (layer.U 1 0)]
  have h11 : layer.U 0 1 ^ 2 + layer.U 1 1 ^ 2 = 1 := by
    have := congr_fun (congr_fun h 1) 1
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply_eq,
               Fin.sum_univ_two] at this
    linarith [sq_abs (layer.U 0 1), sq_abs (layer.U 1 1)]
  have h01 : layer.U 0 0 * layer.U 0 1 + layer.U 1 0 * layer.U 1 1 = 0 := by
    have := congr_fun (congr_fun h 0) 1
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply,
               Fin.sum_univ_two, Fin.zero_eq_one_iff, Nat.succ_ne_self, ↓reduceIte] at this
    linarith
  -- Expand the definition of apply and simplify
  simp only [MERALayer.apply, Matrix.mulVec, Matrix.dotProduct, Fin.sum_univ_two,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
  -- Expand and use orthogonality relations
  have expand : (layer.U 0 0 * x0 + layer.U 0 1 * x1) ^ 2 +
                (layer.U 1 0 * x0 + layer.U 1 1 * x1) ^ 2 =
                (layer.U 0 0 ^ 2 + layer.U 1 0 ^ 2) * x0 ^ 2 +
                (layer.U 0 1 ^ 2 + layer.U 1 1 ^ 2) * x1 ^ 2 +
                2 * (layer.U 0 0 * layer.U 0 1 + layer.U 1 0 * layer.U 1 1) * x0 * x1 := by ring
  rw [expand, h00, h11, h01]
  ring

/-!
### Perfect Reconstruction

Synthesis via U† exactly recovers the original signal:
  ⎡ x_{2k}   ⎤      ⎡ a_k ⎤
  ⎣ x_{2k+1} ⎦ = U† ⎣ d_k ⎦
-/

/-- Perfect reconstruction: U† * (U * x) = x -/
theorem perfect_reconstruction (layer : MERALayer) (x0 x1 : ℝ) :
    let input : Fin 2 → ℝ := ![x0, x1]
    let output := layer.U.mulVec input
    layer.Uᵀ.mulVec output = input := by
  simp only [Matrix.mulVec_mulVec]
  -- Use layer.h_ortho : Uᵀ * U = 1
  rw [layer.h_ortho]
  simp [Matrix.one_mulVec]

/-!
## Special Case: Haar Wavelet (Corollary 1)

The 2-tap QMF filter with constraint h[n] = (-1)ⁿ g[1-n] corresponds
to the Haar wavelet, which is the unique 2-tap orthogonal wavelet.

### QMF Structure for N=2

The QMF constraint h[n] = (-1)ⁿ g[1-n] with N=2 implies:
  h₀ = g₁,  h₁ = -g₀

The polyphase matrix takes the form:
  U = ⎡ g₀   g₁ ⎤
      ⎣ g₁  -g₀ ⎦
-/

/-- Structure of a 2-tap QMF filter bank -/
structure QMFFilterBank where
  g0 : ℝ  -- lowpass coefficient g[0]
  g1 : ℝ  -- lowpass coefficient g[1]
  -- Implicit QMF constraint: h0 = g1, h1 = -g0

/-- Polyphase matrix of a QMF filter bank -/
def QMFFilterBank.toMatrix (qmf : QMFFilterBank) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![qmf.g0, qmf.g1; qmf.g1, -qmf.g0]

/-- Orthogonality condition for QMF: g₀² + g₁² = 1 -/
def QMFFilterBank.isOrthogonal (qmf : QMFFilterBank) : Prop :=
  qmf.g0 ^ 2 + qmf.g1 ^ 2 = 1

/-- Maximum DC gain condition: g₀ = g₁ (both positive) -/
def QMFFilterBank.hasMaxDCGain (qmf : QMFFilterBank) : Prop :=
  qmf.g0 = qmf.g1 ∧ qmf.g0 > 0

/-- Haar QMF filter bank -/
noncomputable def HaarQMF : QMFFilterBank where
  g0 := 1 / Real.sqrt 2
  g1 := 1 / Real.sqrt 2

/-- Lemma: 1/√2 > 0 -/
lemma inv_sqrt2_pos : (1 : ℝ) / Real.sqrt 2 > 0 := by
  apply div_pos one_pos
  exact Real.sqrt_pos.mpr (by norm_num : (2 : ℝ) > 0)

/-- Haar QMF is orthogonal -/
theorem haar_qmf_orthogonal : HaarQMF.isOrthogonal := by
  unfold QMFFilterBank.isOrthogonal HaarQMF
  field_simp

/-- Haar QMF has maximum DC gain -/
theorem haar_qmf_max_dc_gain : HaarQMF.hasMaxDCGain := by
  unfold QMFFilterBank.hasMaxDCGain HaarQMF
  exact ⟨rfl, inv_sqrt2_pos⟩

/-- The QMF matrix is orthogonal iff g₀² + g₁² = 1 -/
theorem qmf_matrix_orthogonal_iff (qmf : QMFFilterBank) :
    qmf.toMatrixᵀ * qmf.toMatrix = 1 ↔ qmf.isOrthogonal := by
  constructor
  · -- (→) Orthogonal matrix implies g₀² + g₁² = 1
    intro h
    unfold QMFFilterBank.isOrthogonal
    have h00 := congr_fun (congr_fun h 0) 0
    simp only [QMFFilterBank.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
               Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
               Matrix.cons_val_one, Matrix.head_cons, Matrix.one_apply_eq] at h00
    linarith [sq_abs qmf.g0, sq_abs qmf.g1]
  · -- (←) g₀² + g₁² = 1 implies orthogonal matrix
    intro hortho
    unfold QMFFilterBank.isOrthogonal at hortho
    ext i j
    simp only [QMFFilterBank.toMatrix, Matrix.mul_apply, Matrix.transpose_apply,
               Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
               Matrix.cons_val_one, Matrix.head_cons, Matrix.one_apply]
    fin_cases i <;> fin_cases j <;> simp [Fin.zero_eq_one_iff, Fin.reduceEq]
    -- Case (0,0): g0*g0 + g1*g1 = 1
    · linarith [sq_abs qmf.g0, sq_abs qmf.g1, hortho]
    -- Case (0,1): g0*g1 + g1*(-g0) = 0
    · ring
    -- Case (1,0): g1*g0 + (-g0)*g1 = 0
    · ring
    -- Case (1,1): g1*g1 + (-g0)*(-g0) = 1
    · linarith [sq_abs qmf.g0, sq_abs qmf.g1, hortho]

/-!
### Corollary 1: Haar Uniqueness

Haar is the unique 2-tap orthogonal wavelet with QMF structure and maximum DC gain.

**Proof:**
1. Orthogonality: g₀² + g₁² = 1
2. Maximum DC gain: g₀ = g₁
3. Substituting: 2g₀² = 1, hence g₀ = 1/√2
4. Therefore: g₀ = g₁ = 1/√2 (Haar)
-/

/-- Corollary 1: Haar Uniqueness.
    If a QMF is orthogonal and has maximum DC gain, then g₀ = g₁ = 1/√2 -/
theorem corollary1_haar_uniqueness (qmf : QMFFilterBank)
    (h_ortho : qmf.isOrthogonal)
    (h_dc : qmf.hasMaxDCGain) :
    qmf.g0 = 1 / Real.sqrt 2 ∧ qmf.g1 = 1 / Real.sqrt 2 := by
  -- Extract hypotheses
  unfold QMFFilterBank.isOrthogonal at h_ortho
  unfold QMFFilterBank.hasMaxDCGain at h_dc
  obtain ⟨h_eq, h_pos⟩ := h_dc
  -- From g₀ = g₁ and g₀² + g₁² = 1, we have 2g₀² = 1
  have h_2g0sq : 2 * qmf.g0 ^ 2 = 1 := by
    calc 2 * qmf.g0 ^ 2 = qmf.g0 ^ 2 + qmf.g0 ^ 2 := by ring
    _ = qmf.g0 ^ 2 + qmf.g1 ^ 2 := by rw [h_eq]
    _ = 1 := h_ortho
  -- Therefore g₀² = 1/2
  have h_g0sq : qmf.g0 ^ 2 = 1 / 2 := by linarith
  -- Since g₀ > 0, we have g₀ = √(1/2) = 1/√2
  have h_g0 : qmf.g0 = 1 / Real.sqrt 2 := by
    have hsqrt : Real.sqrt (1 / 2) = 1 / Real.sqrt 2 := by
      rw [Real.sqrt_div (by norm_num : (1 : ℝ) ≥ 0), Real.sqrt_one]
    rw [← hsqrt]
    exact (Real.sqrt_sq (le_of_lt h_pos)).symm ▸ congrArg Real.sqrt h_g0sq
  -- And g₁ = g₀ = 1/√2
  exact ⟨h_g0, h_eq ▸ h_g0⟩

/-- Normalized Haar matrix -/
noncomputable def HaarMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let c := 1 / Real.sqrt 2
  !![c, c; c, -c]

/-- Haar is orthogonal -/
theorem haar_orthogonal : HaarMatrixᵀ * HaarMatrix = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
  simp only [HaarMatrix, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_two,
    Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one,
    Matrix.one_apply, Fin.zero_eq_one_iff, Fin.reduceEq, ↓reduceIte, one_ne_zero] <;>
  field_simp

/-- The Haar QMF matrix coincides with HaarMatrix -/
theorem haar_qmf_eq_haar_matrix : HaarQMF.toMatrix = HaarMatrix := by
  ext i j
  fin_cases i <;> fin_cases j <;>
  simp only [QMFFilterBank.toMatrix, HaarQMF, HaarMatrix,
             Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
             Matrix.cons_val_one, Matrix.head_cons]

/-- Haar MERA layer -/
noncomputable def HaarMERALayer : MERALayer where
  U := HaarMatrix
  h_ortho := haar_orthogonal

/-!
## Notes on the Formalization

### What is verified:
1. **Theorem 1 (architectural_equivalence)**: MERA ↔ Paraunitary equivalence in time domain
2. **Theorem 1 via Z-domain (theorem1_z_domain)**: Orthogonality ⟺ Paraunitarity
3. **Corollary 1 (corollary1_haar_uniqueness)**: Haar uniqueness
4. Energy conservation (energy_conservation)
5. Perfect reconstruction (perfect_reconstruction)
6. Haar matrix orthogonality (haar_orthogonal)
7. Noble Identities for 2-tap filters (Eq. 10-11, 22-23, 30-31)
8. Power complementarity (|G(ω)|² + |H(ω)|² = 2)
9. Lemma orthogonal_transpose_comm: UᵀU = I ⟹ UUᵀ = I

### Possible extensions:
1. Formalize multi-level decomposition (cascade of L layers)
2. Representation with Laurent polynomial rings ℝ[z, z⁻¹]
3. Wavelets with N > 2 taps (Daubechies, etc.)
-/
