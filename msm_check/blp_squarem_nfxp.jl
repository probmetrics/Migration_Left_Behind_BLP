function fpt_squarem!(delta_fpt, delta_new, delta_init, delta_q1,
					  delta_q2, lnDataShare::AbstractMatrix{T},
					  parm, lnW::AbstractVector{T},
					  lnP::AbstractVector{T}, XQJ_mig::AbstractMatrix{T},
					  XT::AbstractMatrix{T}, XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
					  XF::AbstractMatrix{T}, XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
 					  wgt::AbstractVector{T}, sgwgt::AbstractVector{T},
					  nind::Int, nalt::Int, nsim::Int, dgvec::AbstractVector{Int};
					  alpha::T = 0.12, xdim::Int = 1, ftolerance::T = 1e-14,
					  fpiter::Int = 500, mstep::T = 4.0, stepmin_init::T = 1.0,
					  stepmax_init::T = 1.0, alphaversion::Int = 3) where T <: AbstractFloat
    # default values:
	#	mstem::Float64 = 4.0
    #   stepmin::Float64 = 1.0,
    #   stepmax::Float64 = 1.0,
    #   alphaversion::Int = 3

	delta_old = deepcopy(delta_init)
	## --- update delta ---
	printstyled("BLP Contraction Mapping using SquareM...\n", color = :light_blue)
	j = 1
	deltaconv = one(eltype(ftolerance))
    stepmax = stepmax_init
    stepmin = stepmin_init
	while deltaconv > ftolerance
		if j > fpiter
			printstyled("\nMaximum Fixed Point Iters Reached, NOT Converged\n", color = :light_red)
			break
		end
		## --- begin the SquareM method ---
		update_delta!(delta_new, delta_old, lnDataShare, parm, lnW, lnP, XQJ_mig,
					  XT, XL, XM, XF, XQ, ZSHK, wgt, sgwgt, nind,
					  nalt, nsim, dgvec; alpha = alpha, xdim = xdim)
		@. delta_q1 = delta_new - delta_old
		trace_print(j)
		j += 1

		update_delta!(delta_fpt, delta_new, lnDataShare, parm, lnW, lnP, XQJ_mig,
					  XT, XL, XM, XF, XQ, ZSHK, wgt, sgwgt, nind,
					  nalt, nsim, dgvec; alpha = alpha, xdim = xdim)
		@. delta_q2 = delta_fpt - delta_new
		trace_print(j)
		j += 1

		# -- get the step-size --
		step_alpha = compute_alpha(delta_q1, delta_q2, stepmin, stepmax, alphaversion)
		@. delta_new = delta_old + 2.0 * step_alpha * delta_q1 + step_alpha^2 * (delta_q2 - delta_q1)
		update_delta!(delta_fpt, delta_new, lnDataShare, parm, lnW, lnP, XQJ_mig,
					  XT, XL, XM, XF, XQ, ZSHK, wgt, sgwgt, nind,
					  nalt, nsim, dgvec; alpha = alpha, xdim = xdim)
		trace_print(j)
		j += 1

		# --- error handling ---
		if any(ismissing, delta_fpt) | any(isnan, delta_fpt)
		    warn("Missing values generated during SquareM mapping, switch to BLP mapping\n")
		    update_delta!(delta_new, delta_old, lnDataShare, parm, lnW, lnP, XQJ_mig,
						  XT, XL, XM, XF, XQ, ZSHK, wgt, sgwgt, nind,
						  nalt, nsim, dgvec; alpha = alpha, xdim = xdim)
			j += 1
			trace_print(j)
            update_delta!(delta_fpt, delta_new, lnDataShare, parm, lnW, lnP, XQJ_mig,
						  XT, XL, XM, XF, XQ, ZSHK, wgt, sgwgt, nind,
						  nalt, nsim, dgvec; alpha = alpha, xdim = xdim)
            j += 1
			trace_print(j)
		end

        # -- update step size --
		if step_alpha == stepmax
		    stepmax = mstep * stepmax
		end
		if (step_alpha == stepmin) & (step_alpha < 0)
		    stepmin = mstep * stepmin
		end

		deltaconv = mreldif(delta_fpt, delta_new)
        delta_old = deepcopy(delta_fpt)
	end # <-- end of while loop

    if j <= fpiter
		printstyled("\nContraction Mapping Converged after $j Iterations\n", color = :light_blue)
    end
end

# update_delta!(delta_new, delta_old, lnDataShare, initval, lnW, lnP, XQJ_mig,
# 			  XT, XL, XM, XF, XQ, ZSHK, USHK, wgt, sgwgt, nind, nalt, nsim,
# 			  dgvec; alpha = 0.12, xdim = 1)
function update_delta!(delta_new, delta_old,
					   lnDataShare::AbstractMatrix{T}, parm,
					   lnW::AbstractVector{T}, lnP::AbstractVector{T},
					   XQJ_mig::AbstractMatrix{T},
					   XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					   XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					   XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
					   wgt::AbstractVector{T},
					   sgwgt::AbstractVector{T}, nind::Int, nalt::Int, nsim::Int,
					   dgvec::AbstractVector{Int}; alpha::T = 0.12,
					   xdim::Int = 1) where T <: AbstractFloat

    locpr_thread!(delta_new, parm, delta_old, lnW, lnP, XQJ_mig,
				  XT, XL, XM, XF, XQ, ZSHK, wgt, sgwgt, nind, nalt,
				  nsim, dgvec; alpha = alpha, xdim = xdim)
	broadcast!(log, delta_new, delta_new)
    @fastmath @inbounds @simd for i = eachindex(delta_new)
        delta_new[i] = delta_old[i] + lnDataShare[i] - delta_new[i]
    end
	delta_new[1, :] .= zero(eltype(delta_new))
end

# locpr_thread!(delta_new, initval, delta_old, lnW, lnP, XQJ_mig, XT, XL, XM,
# 			  XF, XQ, ZSHK, USHK, wgt, sgwgt, nind, nalt, nsim, dgvec)
function locpr_thread!(mktshare, parm, Delta, lnW::AbstractVector{T},
					   lnP::AbstractVector{T}, XQJ_mig::AbstractMatrix{T},
					   XT::AbstractMatrix{T},
					   XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
					   XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
					   ZSHK::AbstractMatrix{T},
					   wgt::AbstractVector{T}, sgwgt::AbstractVector{T},
					   nind::Int, nalt::Int, nsim::Int, dgvec::AbstractVector{Int};
					   alpha::T = 0.12, xdim::Int = 1) where T <: AbstractFloat
	##
	## Delta:   nalt x g Matrix
	## XT: 		nt x N Matrix
	## XL: 		nl x N Matrix
	## XM: 		nm x (NJ) Matrix
	## XF: 		nf x (NJ) Matrix
	## XQ: 		nq x N Matrix
	## XQJ_mig: nq x NJ Matrix
	## ZSHK: 	nz x (NS) Matrix
	## USHK:	NS Vector
	## wgt:		N Vector
	## sgwgt:	g Vector, sum of weights for each Delta group
	## dgvec:	N Vector
	##

	bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, =
				unpack_parm(parm, XT, XL, XM, XF, XQ, XQJ_mig, ZSHK, xdim)

	TT = promote_type(eltype(parm), eltype(Delta))
	ngrp = maximum(dgvec)

	# --- begin the loop ---
	mkt_collect = zeros(TT, nalt, ngrp, Threads.nthreads())
	Threads.@threads for k = 1:Threads.nthreads()
		#
		# NOTE: Need to setup containers for each thread !
		#
		xbm = zeros(TT, nalt)
		xbqj_mig = zeros(TT, nalt)
		xbqj_dif = zeros(TT, nalt)
		ln1mlam = zeros(TT, nalt)
		lnq_mig = zeros(TT, nalt)
		dlnq = zeros(TT, nalt)
		zbr = zeros(TT, nsim)

		loc_pri = zeros(TT, nalt)
		tid = Threads.threadid()

		@fastmath @inbounds for i in get_thread_range(nind)
			ind_sel = (1 + nalt * (i - 1)):(i * nalt)
			sim_sel = (1 + nsim * (i - 1)):(i * nsim)
			g = view(dgvec, i)

			loc_prob_ind!(loc_pri, bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, alpha,
						  view(Delta, :, g), xbm, ln1mlam, xbqj_mig, xbqj_dif, dlnq, lnq_mig, zbr,
						  view(lnW, ind_sel), view(lnP, ind_sel), view(XQJ_mig, :, ind_sel),
						  view(XT, :, i), view(XL, :, i),
						  view(XM, :, ind_sel), view(XF, :, ind_sel), view(XQ, :, i),
						  view(ZSHK, :, sim_sel), nalt, nsim)
			BLAS.axpy!(wgt[i], loc_pri, view(mkt_collect, :, g, tid))
		end
	end
	sum!(mktshare, mkt_collect)
	broadcast!(/, mktshare, mktshare, sgwgt')
end

# locpr_serial!(delta_new, initval, delta_old, lnW, lnP, XQJ_mig, XT, XL, XM,
# 			  XF, XQ, ZSHK, USHK, wgt, sgwgt, nind, nalt, nsim, dgvec)
function locpr_serial!(mktshare, parm, Delta, lnW::AbstractVector{T},
						lnP::AbstractVector{T}, XQJ_mig::AbstractMatrix{T},
						XT::AbstractMatrix{T},
						XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
						XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
						ZSHK::AbstractMatrix{T},
						wgt::AbstractVector{T}, sgwgt::AbstractVector{T},
						nind::Int, nalt::Int, nsim::Int, dgvec::AbstractVector{Int};
						alpha::T = 0.12, xdim::Int = 1) where T <: AbstractFloat
	##
	## Delta:   nalt x g Matrix
	## XT: 		nt x N Matrix
	## XL: 		nl x N Matrix
	## XM: 		nm x (NJ) Matrix
	## XF: 		nf x (NJ) Matrix
	## XQ: 		nq x N Matrix
	## XQJ_mig: nq x NJ Matrix
	## ZSHK: 	nz x (NS) Matrix
	## USHK:	NS Vector
	## wgt:		N Vector
	## sgwgt:	g Vector, sum of weights for each Delta group
	## dgvec:	N Vector
	##

	bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, =
		unpack_parm(parm, XT, XL, XM, XF, XQ, XQJ_mig, ZSHK, xdim)

	TT = promote_type(eltype(parm), eltype(Delta))

	# --- setup containers  ---
	xbm = zeros(TT, nalt)
	xbqj_mig = zeros(TT, nalt)
	xbqj_dif = zeros(TT, nalt)
	ln1mlam = zeros(TT, nalt)
	lnq_mig = zeros(TT, nalt)
	dlnq = zeros(TT, nalt)
	zbr = zeros(TT, nsim)
	loc_pri = zeros(TT, nalt)

	# --- begin the loop ---
	@fastmath @inbounds @simd for i = 1:nind
		ind_sel = (1 + nalt * (i - 1)):(i * nalt)
		sim_sel = (1 + nsim * (i - 1)):(i * nsim)
		g = dgvec[i]

		loc_prob_ind!(loc_pri, bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, alpha,
					  view(Delta, :, g), xbm, ln1mlam, xbqj_mig, xbqj_dif, dlnq, lnq_mig, zbr,
					  view(lnW, ind_sel), view(lnP, ind_sel), view(XQJ_mig, :, ind_sel),
					  view(XT, :, i), view(XL, :, i),
					  view(XM, :, ind_sel), view(XF, :, ind_sel), view(XQ, :, i),
					  view(ZSHK, :, sim_sel), nalt, nsim)
		BLAS.axpy!(wgt[i], loc_pri, view(mktshare, :, g))
	end
	broadcast!(/, mktshare, mktshare, sgwgt')
end

using StatsFuns:logistic, log1pexp
function loc_prob_ind!(loc_pri, bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz,
						alpha, delta, xbm, ln1mlam, xbqj_mig, xbqj_dif, dlnq, lnq_mig, zbr, lnw,
						lnp, xqj_mig, xt, xl, xm, xf, xq, zshk, nalt, nsim)
	##
	## delta: 		J x 1 Vector
	## lnw, lnp: 	J x 1 Vector
	## xbt, xbl: 	scalar
	## xbm: 		J x 1 Vector
	## ln1mlam, 	J x 1 Vector
	## xbq, 		scalar or J Vector
	## zbr, 		S x 1 Vector
	## ushk:		S x 1 Vector
	##

	# --- intermediate vars ---
	xbt = xt' * bt
	xbl = xl' * bl
	xbq = xq' * bq
	mul!(xbm, xm', bm)
	mul!(ln1mlam, xf', bf)
	broadcast!(nlog1pexp, ln1mlam, ln1mlam)

	mul!(xbqj_mig, xqj_mig', bqj_mig)
	mul!(xbqj_dif, xqj_mig', bqj_dif)
	lnq_alt!(lnq_mig, dlnq, lnw, ln1mlam, xbq, xbqj_mig, xbqj_dif, bw, blft, bitr)
	mul!(zbr, zshk', bz)

	# --- setup containers ---
	TT = promote_type(eltype(bw), eltype(lnw))
	unit = one(TT)
	loc_pr_s = zeros(TT, nalt)
	fill!(loc_pri, zero(TT))

	# --- begin the loop ---
	@fastmath @inbounds @simd for s = 1:nsim
		zrnd = zbr[s]
		theta = logistic(xbt + zrnd)
		for j = 1:nalt
			# calculate location prob
			gambar = gamfun(lnw[j], dlnq[j], lnq_mig[j], xbl, ln1mlam[j], theta)

			# --- location specific utility ---
			loc_pr_s[j] = Vloc(alpha, lnp[j], theta, xbm[j], gambar, delta[j])
		end
		emaxprob!(loc_pr_s)
		# softmax!(loc_pr_s, loc_pr_s)
		loc_pri .+= loc_pr_s
	end #<-- end of s loop

	return loc_pri ./= nsim
end

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
