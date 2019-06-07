function qfun(lnw, xbq, xbqc, ln1mlam, bw, blft, bitr)
    # blft should be negative
    dlnq = blft + bitr * lnw - bw * ln1mlam - xbqc
    lnq_lft = blft + bw * lnw + bitr * lnw + xbq
    lnq_mig = lnq_lft - dlnq
    return (dlnq, lnq_mig, lnq_lft)
end

using StatsFuns:logaddexp
function gamfun(lnw, dlnq, lnq_mig, xbl, ln1mlam, theta, psi)
    Vlft = theta * dlnq - xbl
    Vmig = psi * ln1mlam
    gambar = logaddexp(Vlft, Vmig) + psi * lnw + theta * lnq_mig
    return gambar
end

using StatsFuns:logistic
function leftbh_prob(theta, psi, ln1mlam, xbl, dlnq)
    dVleft = -psi * ln1mlam - xbl + theta * dlnq
    lftpr = logistic(dVleft)
    return lftpr
end

function Vloc(xbm, gambar, delta)
    Vloc = gambar - xbm + delta
    return Vloc
end

using StatsFuns:log1pexp
function neglog1pexp(x::T) where T <: Real
    return -log1pexp(x)
end

function gsel_range(g::Int, nalt::Int)
    ans = (1 + nalt * (g - 1)):(g * nalt)
    return ans
end

function locate_gidx(x::Int, ngvec::AbstractVector{Int})
    g = 1
    for v in ngvec
        g += 1 - (x <= v)
    end
    return g
end
