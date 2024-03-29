function lnq_alt!(lnq_mig, dlnq, lnw, ln1mlam, xbq, xbqj_mig, xbqj_dif,
				  bw, blft, bitr)
	for j = 1:length(lnq_mig)
		dlnq_j, lnq_mig_j = qfun(lnw[j], ln1mlam[j], xbq, xbqj_mig[j],
								 xbqj_dif[j], bw, blft, bitr)
		lnq_mig[j] = lnq_mig_j
		dlnq[j] = dlnq_j
	end
end

function qfun(lnw, ln1mlam, xbq, xbqj_mig, xbqj_dif, bw, blft, bitr)
    # b2 = blft < 0
    # b1 = bw > 0
    # b3 = bitr < 0
    dlnq = blft + bitr * lnw - bw * ln1mlam + xbqj_dif
	lnq_mig = bw * lnw + bw * ln1mlam + xbq + xbqj_mig
    return (dlnq, lnq_mig)
end

using StatsFuns:logaddexp
function gamfun(lnw, dlnq, lnq_mig, xbl, ln1mlam, theta)
	unit = oneunit(theta)
    Vlft = (unit - theta) * dlnq - xbl
    Vmig = theta * ln1mlam
    gambar = logaddexp(Vlft, Vmig) + theta * lnw + (unit - theta) * lnq_mig
    return gambar
end

using StatsFuns:logistic
function leftbh_prob(theta, ln1mlam, xbl, dlnq)
    dVlft = - theta * ln1mlam - xbl + (oneunit(theta) - theta) * dlnq
    lftpr = logistic(dVlft)
    return lftpr
end

function Vloc(alpha, lnp, theta, xbm, gambar, delta)
    Vloc = gambar - alpha * theta * lnp - xbm + delta
    return Vloc
end

using StatsFuns:log1pexp
function nlog1pexp(x::T) where T <: Real
    outx = - log1pexp(x)
    return outx
end

using Sobol
using StatsFuns:norminvcdf
function draw_shock(nsim::Integer; dims::Integer = 1, npcut::Integer = 53)
    sbseq = SobolSeq(dims)
    sbgrids = norminvcdf.(hcat([next!(sbseq) for i = 1:(nsim + npcut)]...)'[(npcut + 1):end, :])
    return sbgrids
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

# function mydot(x::AbstractArray{T}, y::AbstractArray{T}) where T <: Real
# 	size(x) == size(y) || error("x and y should have the same size")
#     s = zero(eltype(x))
#     @inbounds for i in eachindex(x)
# 		s += x[i] * y[i]
#     end
# 	return s
# end

using Random, DataFrames
using StatsBase:sample
function boot_df(data::AbstractDataFrame, vnames::Union{Symbol, Array{Symbol, 1}};
                   nboot::Integer = 100)
    # vnames = [:caring_study, :parenting_att, :cedu_expect]
    # joint sampling of vnames!
    if typeof(vnames) == Symbol
        vnames = [vnames]
    end
    all(v -> in(v, names(data)), vnames) || error("Not all vnames in data")

    sel = prod(mapreduce(x -> .!ismissing.(data[:, x]), hcat, vnames), dims = 2)
    selpos = (LinearIndices(sel))[findall(sel)]

    bootsel = sample(selpos, nboot; replace = true)
	out = data[bootsel, vnames]
    # out = convert(Array{Float64, 2}, data[bootsel, vnames])
    return out
end

using Random, DataFrames
using StatsBase:sample
function boot_df_by(data::AbstractDataFrame, vnames::Union{Symbol, Array{Symbol, 1}};
				 	byvars::Union{Nothing, Symbol, Array{Symbol, 1}} = nothing,
                 	multiplier::Real = 1.0)
    if typeof(vnames) == Symbol
        vnames = [vnames]
    end
    all(v -> in(v, names(data)), vnames) || error("Not all vnames in data")

    if byvars == nothing
		nboot = Int(floor(nrow(data) * multiplier))

		sel = prod(mapreduce(x -> .!ismissing.(data[:, x]), hcat, vnames), dims = 2)
    	selpos = (LinearIndices(sel))[findall(sel)]

		bootsel = sample(selpos, nboot; replace = true)
        outdata = data[bootsel, vnames]
    else
        if typeof(byvars) == Symbol
            byvars = [byvars]
        end

        all(v -> in(v, names(data)), byvars) || error("Not all byvars in data")

		outdata = by(data, byvars) do df
            nboot = Int(floor(nrow(df) * multiplier))

			sel = prod(mapreduce(x -> .!ismissing.(df[:, x]), hcat, vnames), dims = 2)
    		selpos = (LinearIndices(sel))[findall(sel)]

            bootsel = sample(selpos, nboot; replace = true)
            df[bootsel, vnames]
        end
    end
    return outdata
end
