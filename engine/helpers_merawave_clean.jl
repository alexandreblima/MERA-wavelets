using LinearAlgebra
using TensorOperations

# Utilities
function Square(T)
    @tensor T2[a,b] := T[a,c] * T[c,b]
    return T2
end

function normalize_isometry(w)
    chi, d1, d2 = size(w)
    wf_re = reshape(w, chi, d1 * d2)
    U, _, V = svd(wf_re)
    return reshape(U * V', chi, d1, d2)
end

# Superoperators
function AscendSuper(op, u, w)
    @tensor begin
        tmp1[l, r, il, ir] := u[l, r, cl, cr] * op[cl, cr, il, ir]
        tmp2[l, r, cl, cr] := tmp1[l, r, il, ir] * conj(u[cl, cr, il, ir])
        op_asc[ol, or, il, ir] := w[ol, ll, lr] * tmp2[ll, rl, il, ir] * w[or, rl, rr] * conj(w[il, cl, cr]) * conj(w[ir, cl, cr])
    end
    return op_asc
end

function DescendSuper(rho, u, w)
    @tensor begin
        tmp1[il, ir, ol, or, ll, lr, rl, rr] := conj(w[il, ll, lr]) * rho[il, ir, ol, or] * w[ir, rl, rr]
        tmp2[il, ir, ol, or, cl, cr] := tmp1[il, ir, ol, or, ll, lr, rl, rr] * conj(w[ir, cl, cr])
        rho_desc[ol, or, il, ir] := conj(u[ol, or, cl, cr]) * tmp2[il, ir, ol, or, il, ir]
    end
    return rho_desc
end

# Tensor update (Evenbly)
function TensorUpdateSVD_Evenbly(A, B)
    sA = size(A); sB = size(B)
    @tensor M[al, bl, cl, dl] := A[al, ar, ac, ad] * B[bl, br, bc, bd]
    U, _, V = svd(reshape(M, sA[1]*sB[1], sA[3]*sB[3]))
    return reshape(U * V', sA[1], sB[1], sA[3], sB[3])
end
const TensorUpdateSVD = TensorUpdateSVD_Evenbly

# Environments
function DisentanglerEnv(rho, op, w)
    @tensor begin
        A[ol, or, il, ir] := rho[ol, or, cl, cr] * conj(w[cl, il, al]) * conj(w[cr, ir, ar])
        B[ol, or, il, ir] := op[cl, cr, il, ir] * w[cl, ol, al] * w[cr, or, ar]
    end
    return A, B
end

function IsometryEnv(rho, op, u)
    @tensor begin
        A[o, l, r] := conj(u[cl, cr, l, r]) * rho[il, ir, cl, cr] * conj(u[il, ir, al, ar])
        B[o, l, r] := u[il, ir, l, r] * op[cl, cr, il, ir] * u[cl, cr, al, ar]
    end
    return A, B
end
