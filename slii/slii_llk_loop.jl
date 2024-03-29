function mig_leftbh_llk(parm, Delta::AbstractMatrix{T}, YL::AbstractVector{T},
						YM::AbstractVector{T}, lnW::AbstractVector{T},
						lnP::AbstractVector{T}, lnQbar::AbstractVector{T},
						XQJ_mig::AbstractMatrix{T},	XT::AbstractMatrix{T},
						XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
						XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
						ZSHK::AbstractMatrix{T}, USHK::AbstractVector{T},
						QSHK::AbstractVector{T}, wgt::AbstractVector{T}, nind::Int,
						nalt::Int, nsim::Int, dgvec::AbstractVector{Int},
						dage9vec::AbstractVector{T}, lnqrsd::T, alpha::T,
						xdim::Int) where T <: AbstractFloat
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
	## dgvec:	N Vector
	##

	bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, sigu, sigq =
		unpack_slii_parm(parm, XT, XL, XM, XF, XQ, XQJ_mig, ZSHK, xdim)

	# --- setup containers ---
	TT = promote_type(eltype(parm), eltype(Delta))
	xbm = zeros(TT, nalt)
	xbqj_mig = zeros(TT, nalt)
	xbqj_dif = zeros(TT, nalt)
	ln1mlam = zeros(TT, nalt)
	lnq_mig = zeros(TT, nalt)
	lnq_lft = zeros(TT, nalt)
	dlnq = zeros(TT, nalt)
	zbr = zeros(TT, nsim)

	# --- begin the loop ---
	llk = zero(TT)
	@fastmath @inbounds @simd for i = 1:nind
		ind_sel = (1 + nalt * (i - 1)):(i * nalt)
		sim_sel = (1 + nsim * (i - 1)):(i * nsim)
		g = view(dgvec, i)
		dage9 = dage9vec[i]

		llk += individual_llk(bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif,
							  bz, sigu, sigq, lnqrsd, alpha, view(Delta, :, g),
							  xbm, ln1mlam, xbqj_mig, xbqj_dif, dlnq, lnq_mig, lnq_lft, zbr,
							  view(YL, ind_sel), view(YM, ind_sel), view(lnW, ind_sel),
							  view(lnP, ind_sel), view(lnQbar, ind_sel), view(XQJ_mig, :, ind_sel),
							  view(XT, :, i), view(XL, :, i), view(XM, :, ind_sel),
							  view(XF, :, ind_sel), view(XQ, :, i), dage9, view(ZSHK, :, sim_sel),
							  view(USHK, sim_sel), view(QSHK, sim_sel), nalt, nsim) * wgt[i]
	end

	return llk
end


using StatsFuns:logistic, log1pexp, normpdf
function individual_llk(bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, sigu,
						sigq, lnqrsd, alpha, delta, xbm, ln1mlam, xbqj_mig, xbqj_dif,
						dlnq, lnq_mig, lnq_lft, zbr, yl, ym, lnw, lnp, lnqbar, xqj_mig, xt,
						xl, xm, xf, xq, dage9, zshk, ushk, qshk, nalt, nsim)
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
	broadcast!(+, lnq_lft, lnq_mig, dlnq)
	mul!(zbr, zshk', bz)

	# --- setup containers ---
	TT = promote_type(eltype(bw), eltype(lnw))
	unit = one(TT)
	llki = zero(TT)

	# --- begin the loop ---
	@fastmath @inbounds @simd for s = 1:nsim
		qllk_js = zero(TT)
		lftpr_s = zero(TT)
		eVjs = zero(TT)
		eVsums = zero(TT)

		zrnd = zbr[s]
		urnd = ushk[s]
		qrnd = qshk[s]
		theta = logistic(xbt + zrnd - sigu * urnd)
		lnqrnd = lnqrsd * qrnd

		for j = 1:nalt
			# --- likelihood of lnq ---
			lnq_js = (lnq_lft[j] + lnqrnd) * ym[j] + (lnq_mig[j] + lnqrnd) * (unit - ym[j])
			qllk_js += normpdf(lnqbar[j], sigq, lnq_js) * yl[j]

			# --- calculate lft_prob ---
			lft_pr_tmp = leftbh_prob(theta, ln1mlam[j], xbl, dlnq[j])
			lftpr_s += (lft_pr_tmp * ym[j] + (unit - ym[j]) * (unit - lft_pr_tmp)) * yl[j]

			# --- calculate location prob ---
			gambar = gamfun(lnw[j], dlnq[j], lnq_mig[j], xbl, ln1mlam[j], theta)

			# --- location specific utility ---
			eVj = exp(Vloc(alpha, lnp[j], theta, xbm[j], gambar, delta[j]))
			eVjs += eVj * yl[j]
			eVsums += eVj
		end
		llki += (eVjs / eVsums) * lftpr_s * (qllk_js * dage9 + unit - dage9)

	end #<-- end of s loop

	return -log(llki / nsim)
end

function unpack_slii_parm(parm, XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					 XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					 XQ::AbstractMatrix{T}, XQJ_mig::AbstractMatrix{T},
					 ZSHK::AbstractMatrix{T}, xdim::Int) where T <: AbstractFloat
	 nxt = size(XT, xdim)
	 nxl = size(XL, xdim)
	 nxm = size(XM, xdim)
	 nxf = size(XF, xdim)
	 nxq = size(XQ, xdim)
	 nxqj = size(XQJ_mig, xdim)
	 nzr = size(ZSHK, xdim)

	 bt = parm[1:nxt]
	 bl = parm[(nxt + 1):(nxt + nxl)]
	 bm = parm[(nxt + nxl + 1):(nxt + nxl + nxm)]
	 bf = parm[(nxt + nxl + nxm + 1):(nxt + nxl + nxm + nxf)]

	 blft = parm[nxt + nxl + nxm + nxf + 1]
	 bw = parm[nxt + nxl + nxm + nxf + 2]
	 bitr = parm[nxt + nxl + nxm + nxf + 3]

	 bq = parm[(nxt + nxl + nxm + nxf + 4):(nxt + nxl + nxm + nxf + nxq + 3)]
	 bqj_mig = parm[(nxt + nxl + nxm + nxf + nxq + 4):(nxt + nxl + nxm + nxf + nxq + nxqj + 3)]
	 bqj_dif = parm[(nxt + nxl + nxm + nxf + nxq + nxqj + 4):(nxt + nxl + nxm + nxf + nxq + 2*nxqj + 3)]

	 bz = parm[(nxt + nxl + nxm + nxf + nxq + 2*nxqj + 4):(nxt + nxl + nxm + nxf + nxq + 2*nxqj + nzr + 3)] #<- observed household char.
	 sigu = exp(parm[nxt + nxl + nxm + nxf + nxq + 2*nxqj + nzr + 4])
	 sigq = exp(parm[nxt + nxl + nxm + nxf + nxq + 2*nxqj + nzr + 5])
	 return (bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, sigu, sigq)
end

# function tadd(x::AbstractVector{T}) where T <: AbstractFloat
# 	xlen = length(x)
# 	s = Threads.Atomic{eltype(x)}(0)
# 	Threads.@threads for i = 1:xlen
#        Threads.atomic_add!(s, x[i])
#     end
# 	return s[]
# end

function mig_leftbh_llk_thread(parm, Delta::AbstractMatrix{T}, YL::AbstractVector{T},
						YM::AbstractVector{T}, lnW::AbstractVector{T},
						lnP::AbstractVector{T}, lnQbar::AbstractVector{T},
						XQJ_mig::AbstractMatrix{T},	XT::AbstractMatrix{T},
						XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
						XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
						ZSHK::AbstractMatrix{T}, USHK::AbstractVector{T},
						QSHK::AbstractVector{T}, wgt::AbstractVector{T}, nind::Int,
						nalt::Int, nsim::Int, dgvec::AbstractVector{Int},
						dage9vec::AbstractVector{T}, lnqrsd::T, alpha::T,
						xdim::Int) where T <: AbstractFloat
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
	## dgvec:	N Vector
	##

	bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, sigu, sigq =
		unpack_slii_parm(parm, XT, XL, XM, XF, XQ, XQJ_mig, ZSHK, xdim)

	TT = promote_type(eltype(parm), eltype(Delta))

	# --- begin the loop ---
	llk = zeros(TT, Threads.nthreads())
	Threads.@threads for k = 1:Threads.nthreads()
		#
		# NOTE: Need to setup containers for each thread !
		#
		llk_thread = zero(TT)
		xbm = zeros(TT, nalt)
		xbqj_mig = zeros(TT, nalt)
		xbqj_dif = zeros(TT, nalt)
		ln1mlam = zeros(TT, nalt)
		lnq_mig = zeros(TT, nalt)
		lnq_lft = zeros(TT, nalt)
		dlnq = zeros(TT, nalt)
		zbr = zeros(TT, nsim)

		@fastmath @inbounds for i in get_thread_range(nind)
			ind_sel = (1 + nalt * (i - 1)):(i * nalt)
			sim_sel = (1 + nsim * (i - 1)):(i * nsim)
			g = view(dgvec, i)
			dage9 = dage9vec[i]

			llk_thread += individual_llk(bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif,
										 bz, sigu, sigq, lnqrsd, alpha, view(Delta, :, g), xbm, ln1mlam,
										 xbqj_mig, xbqj_dif, dlnq, lnq_mig, lnq_lft, zbr, view(YL, ind_sel),
								  		 view(YM, ind_sel), view(lnW, ind_sel),  view(lnP, ind_sel),
								 	 	 view(lnQbar, ind_sel), view(XQJ_mig, :, ind_sel), view(XT, :, i),
										 view(XL, :, i), view(XM, :, ind_sel), view(XF, :, ind_sel),
								  		 view(XQ, :, i), dage9, view(ZSHK, :, sim_sel), view(USHK, sim_sel),
										 view(QSHK, sim_sel), nalt, nsim) * wgt[i]
		end
		llk[Threads.threadid()] = llk_thread
	end
	return sum(llk)
end

function get_thread_range(n)
    tid = Threads.threadid()
    nt = Threads.nthreads()
    d , r = divrem(n, nt)
    from = (tid - 1) * d + min(r, tid - 1) + 1
    to = from + d - 1 + (tid <= r ? 1 : 0)
    from:to
end
