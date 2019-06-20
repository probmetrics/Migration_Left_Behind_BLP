function locpr_thread!(mktshare, parm, Delta::AbstractMatrix{T}, lnW::AbstractVector{T},
					   lnP::AbstractVector{T}, XT::AbstractMatrix{T},
					   XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
					   XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
					   ZSHK::AbstractMatrix{T}, USHK::AbstractVector{T},
					   wgt::AbstractVector{T}, sgwgt::AbstractVector{T},
					   nind::Int, nalt::Int, nsim::Int, dgvec::AbstractVector{Int};
					   alpha::AbstractFloat = 0.12, xdim::Int = 1) where T <: AbstractFloat
	##
	## Delta:   nalt x g Matrix
	## XT: 		nt x N Matrix
	## XL: 		nl x N Matrix
	## XM: 		nm x (NJ) Matrix
	## XF: 		nf x (NJ) Matrix
	## XQ: 		nq x NJ Matrix
	## ZSHK: 	nz x (NS) Matrix
	## USHK:	NS Vector
	## wgt:		N Vector
	## sgwgt:	g Vector, sum of weights for each Delta group
	## dgvec:	N Vector
	##

	bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu = unpack_parm(parm, XT, XL, XM, XF, XQ, ZSHK; xdim = xdim)

	TT = promote_type(eltype(parm), eltype(Delta_init))
	ngrp = maximum(dgvec)

	# --- begin the loop ---
	mkt_collect = zeros(TT, nalt, ngrp, Threads.nthreads())
	Threads.@threads for k = 1:Threads.nthreads()
		#
		# NOTE: Need to setup containers for each thread !
		#
		xbm = zeros(TT, nalt)
		xbq = zeros(TT, nalt)
		ln1mlam = zeros(TT, nalt)
		lnq_mig = zeros(TT, nalt)
		dlnq = zeros(TT, nalt)
		zbr = zeros(TT, nsim)

		loc_pri = zeros(TT, nalt)
		tid = Threads.threadid()

		@fastmath @inbounds for i in getrange(nind)
			ind_sel = (1 + nalt * (i - 1)):(i * nalt)
			sim_sel = (1 + nsim * (i - 1)):(i * nsim)
			g = view(dgvec, i)

			loc_prob_ind!(loc_pri, bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu, alpha,
						  view(Delta, :, g), xbm, ln1mlam, xbq, dlnq, lnq_mig, zbr,
						  view(lnW, ind_sel), view(lnP, ind_sel), view(XT, :, i), view(XL, :, i),
						  view(XM, :, ind_sel), view(XF, :, ind_sel), view(XQ, :, ind_sel),
						  view(ZSHK, :, sim_sel), view(USHK, sim_sel), nalt, nsim)
			BLAS.axpy!(wgt[i], loc_pri, view(mkt_collect, :, g, tid))
		end
	end
	sum!(mktshare, mkt_collect)
	broadcast!(/, mktshare, mktshare, sgwgt')
end

function locpr_serial!(mktshare, parm, Delta::AbstractMatrix{T}, lnW::AbstractVector{T},
						lnP::AbstractVector{T}, XT::AbstractMatrix{T},
						XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
						XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
						ZSHK::AbstractMatrix{T}, USHK::AbstractVector{T},
						wgt::AbstractVector{T}, sgwgt::AbstractVector{T},
						nind::Int, nalt::Int, nsim::Int, dgvec::AbstractVector{Int};
						alpha::AbstractFloat = 0.12, xdim::Int = 1) where T <: AbstractFloat
	##
	## Delta:   nalt x g Matrix
	## XT: 		nt x N Matrix
	## XL: 		nl x N Matrix
	## XM: 		nm x (NJ) Matrix
	## XF: 		nf x (NJ) Matrix
	## XQ: 		nq x NJ Matrix
	## ZSHK: 	nz x (NS) Matrix
	## USHK:	NS Vector
	## wgt:		N Vector
	## sgwgt:	g Vector, sum of weights for each Delta group
	## dgvec:	N Vector
	##

	bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu = unpack_parm(parm, XT, XL, XM, XF, XQ, ZSHK; xdim = xdim)

	TT = promote_type(eltype(parm), eltype(Delta_init))

	# --- setup containers  ---
	xbm = zeros(TT, nalt)
	xbq = zeros(TT, nalt)
	ln1mlam = zeros(TT, nalt)
	lnq_mig = zeros(TT, nalt)
	dlnq = zeros(TT, nalt)
	zbr = zeros(TT, nsim)
	loc_pri = zeros(TT, nalt)

	# --- begin the loop ---
	@fastmath @inbounds @simd for i = 1:nind
		ind_sel = (1 + nalt * (i - 1)):(i * nalt)
		sim_sel = (1 + nsim * (i - 1)):(i * nsim)
		g = view(dgvec, i)

		loc_prob_ind!(loc_pri, bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu, alpha,
					  view(Delta, :, g), xbm, ln1mlam, xbq, dlnq, lnq_mig, zbr,
					  view(lnW, ind_sel), view(lnP, ind_sel), view(XT, :, i), view(XL, :, i),
					  view(XM, :, ind_sel), view(XF, :, ind_sel), view(XQ, :, ind_sel),
					  view(ZSHK, :, sim_sel), view(USHK, sim_sel), nalt, nsim)
		BLAS.axpy!(wgt[i], loc_pri, view(mktshare, :, g))
	end
	broadcast!(/, mktshare, mktshare, sgwgt')
end

using StatsFuns:logistic, log1pexp, softmax!
function loc_prob_ind!(loc_pri, bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu, alpha, delta,
						xbm, ln1mlam, xbq, dlnq, lnq_mig, zbr, lnw, lnp, xt,
						xl, xm, xf, xq, zshk, ushk, nalt, nsim)
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
	mul!(xbm, xm', bm)
	mul!(ln1mlam, xf', bf)
	broadcast!(log1pexp, ln1mlam, ln1mlam)

	mul!(xbq, xq', bq)
	lnq_alt!(lnq_mig, dlnq, lnw, ln1mlam, xbq, bw, blft, bitr)
	mul!(zbr, zshk', bz)

	# --- setup containers ---
	TT = promote_type(eltype(bw), eltype(lnw))
	unit = one(TT)
	loc_pr_s = zeros(TT, nalt)
	fill!(loc_pri, zero(TT))

	# --- begin the loop ---
	@fastmath @inbounds @simd for s = 1:nsim
		zrnd = zbr[s]
		urnd = ushk[s]
		theta = logistic(xbt + zrnd + sigu * urnd)
		for j = 1:nalt
			# calculate location prob
			gambar = gamfun(lnw[j], dlnq[j], lnq_mig[j], xbl, ln1mlam[j], theta)

			# --- location specific utility ---
			loc_pr_s[j] = Vloc(alpha, lnp[j], theta, xbm[j], gambar, delta[j])
		end
		softmax!(loc_pr_s, loc_pr_s)
		loc_pri .+= loc_pr_s
	end #<-- end of s loop

	return loc_pri ./= nsim
end
