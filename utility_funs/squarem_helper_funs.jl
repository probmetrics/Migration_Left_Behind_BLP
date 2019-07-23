function mreldif(x::AbstractArray{T}, y::AbstractArray{T}) where T <: Real
    size(x) == size(y) || error("size of two Arrays should be identical")
	TT = eltype(x)
    m = zero(TT)
    @fastmath @inbounds @simd for i = eachindex(x)
        m = max(m, abs(x[i] - y[i]) / (abs(x[i]) + one(TT)))
    end
    return m
end

function compute_alpha(delta_q1::AbstractArray, delta_q2::AbstractArray,
                        stepmin::T, stepmax::T, alphaversion::Int) where T <: AbstractFloat
    # default values:
    #   stepmin::Float64 = 1.0,
    #   stepmax::Float64 = 1.0,
    #   alphaversion::Int = 3
	TT = eltype(delta_q1)
    sr2 = zero(TT)
    sq2 = zero(TT)
    sv2 = zero(TT)
    srv = zero(TT)
    @fastmath @inbounds @simd for i = eachindex(delta_q1)
        sr2 += delta_q1[i]^2
        sq2 += delta_q2[i]^2
        sv2 += (delta_q2[i] - delta_q1[i])^2
        srv += delta_q1[i] * (delta_q2[i] - delta_q1[i])
    end
    sq2 = sqrt(sq2)

	if alphaversion == 1
        step_alpha = -srv / sv2
	elseif alphaversion == 2
        step_alpha = -sr2 / srv
	else
        step_alpha = sqrt(sr2 / sv2)
    end

	step_alpha = max(stepmin, min(stepmax, step_alpha))
	return step_alpha
end

function trace_print(j::Int; line_end::Int = 36)
    io = repeat("-", j % line_end)
        print(io, "\u1b[1G") # go to first column
    if j % line_end == 0
        printstyled("\n", io, ">$j", color = :light_blue)
        # print_with_color(:light_blue, "\n", io, ">$j")
    else
        printstyled(io, ">$j", color = :light_blue)
        # print_with_color(:light_blue, io, ">$j")
    end
    print("", "\u1b[K") # clear the rest of the line
end
