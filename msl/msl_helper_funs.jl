function lnq_alt!(lnq_mig, dlnq, lnw, ln1mlam, bw, blft, bitr)
	for j = 1:length(lnq_mig)
		dlnq_j, lnq_mig_j = qfun(lnw[j], ln1mlam[j], bw, blft, bitr)
		lnq_mig[j] .= lnq_mig_j
		dlnq[j] .= dlnq_j
	end
end

function qfun(lnw, ln1mlam, xbq, bw, blft, bitr)
    # b2 = blft < 0
    # b1 = bw > 0
    # b3 = bitr < 0
    dlnq = blft + bitr * lnw + bw * ln1mlam
	lnq_mig = bw * lnw - bw * ln1mlam + xbq
    return (dlnq, lnq_mig)
end

using StatsFuns:logaddexp
function gamfun(lnw, dlnq, lnq_mig, xbl, ln1mlam, theta)
    Vlft = (1.0 - theta) * dlnq - xbl
    Vmig = - theta * ln1mlam
    gambar = logaddexp(Vlft, Vmig) + theta * lnw + (1.0 - theta) * lnq_mig
    return gambar
end

using StatsFuns:logistic
function leftbh_prob(theta, ln1mlam, xbl, dlnq)
    dVlft = theta * ln1mlam - xbl + (1.0 - theta) * dlnq
    lftpr = logistic(dVlft)
    return lftpr
end

function Vloc(alpha, lnp, theta, xbm, gambar, delta)
    Vloc = gambar - alpha * theta * lnp - xbm + delta
    return Vloc
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

function emaxprob!(x::AbstractArray{T}) where T <: Real
    n = length(x)
    u = maximum(x)
    s = zero(eltype(x))
    @inbounds for i = 1:n
        x[i] = exp(x[i] - u)
		s += x[i]
    end
	x ./= s
end
