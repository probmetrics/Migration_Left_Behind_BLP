function msm_obj(parm, data_mnts::AbstractVector{T}, dwt::AbstractVector{T},
				 alpha::T, lnW::AbstractVector{T}, lnP::AbstractVector{T},
				 XQJ_mig::AbstractVector{T}, XQJ_lft::AbstractVector{T},
				 XT::AbstractMatrix{T}, XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
				 XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
				 ZSHK::AbstractMatrix{T}, USHK::AbstractVector{T}, QSHK::AbstractVector{T},
				 pr_lft::AbstractVector{T}, pr_lft_alt::AbstractMatrix{T},
				 Delta::AbstractMatrix{T}, dgvec::AbstractVector{Int},
				 htvec::AbstractVector{Int}, dage9vec::AbstractVector{T},
				 wgt::AbstractVector{T}, sgwgt::AbstractVector{T},
				 swgt9::T, nind::Int, nalt::Int,
				 nsim::Int; xdim::Int = 1) where T <: AbstractFloat

	mnt_par = get_moments_thread(parm, alpha, lnW, lnP, XQJ_mig, XQJ_lft, XT, XL, XM, XF, XQ,
								 ZSHK, USHK, QSHK, pr_lft, pr_lft_alt, Delta_init,
								 dgvec, htvec, dage9vec, wgt, sgwgt, swgt9, nind,
								 nalt, nsim; xdim = xdim)
	gmnt = mnt_par - data_mnts
	broadcast!(*, gmnt, gmnt, dwt, gmnt)
	obj = sum(gmnt)
 	return obj
end

function get_moments_thread(parm, alpha::T, lnW::AbstractVector{T},
					 lnP::AbstractVector{T}, XQJ_mig::AbstractVector{T},
					 XQJ_lft::AbstractVector{T},
					 XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					 XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					 XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
					 USHK::AbstractVector{T}, QSHK::AbstractVector{T},
					 pr_lft::AbstractVector{T}, pr_lft_alt::AbstractMatrix{T},
					 Delta::AbstractMatrix{T}, dgvec::AbstractVector{Int},
					 htvec::AbstractVector{Int}, dage9vec::AbstractVector{T},
					 wgt::AbstractVector{T}, sgwgt::AbstractVector{T},
					 swgt9::T, nind::Int, nalt::Int,
					 nsim::Int; xdim::Int = 1) where T <: AbstractFloat

	bw, blft, bitr, bqxj, bt, bl, bm, bf, bq, bz, sigu,
	rhoq, sigq, mnt_idx, mnt_drop, mnt_cage9 =
			unpack_parm_msm(parm, XT, XL, XM, XF, XQ, ZSHK, nalt, xdim)

	# --- common variables ---
	TT = promote_type(eltype(parm), eltype(lnW))
	unit = one(TT)
	mnt_len = sum(mnt_idx)
	mnt_range = get_mnt_range(mnt_idx)
	ht_len = length(unique(htvec))

	mntmat_all = zeros(TT, mnt_len, ht_len, Threads.nthreads())
	mntmat = zeros(TT, mnt_len, ht_len)
	Threads.@threads for k = 1:Threads.nthreads()
		##
		## NOTE: setup containers for each thread!
		##
		mntvec_thread = zeros(TT, mnt_len)
		mntmat_thread = zeros(TT, mnt_len, ht_len)

		mktshare = zeros(TT, nalt)
		lftshare = zeros(TT, nalt)
		lftpr_is = zeros(TT, nalt)
		migpr_is = zeros(TT, nalt)
		locpr_is = zeros(TT, nalt)
		lnqrnd_lftj = zeros(TT, nalt)
		lnqrnd_migj = zeros(TT, nalt)
		nalt_tmp = zeros(TT, nalt)
		xf_xt_p = zeros(TT, length(bf), length(bt))

		xbm = zeros(TT, nalt)
		xbq = zeros(TT, nalt)
		ln1mlam = zeros(TT, nalt)
		lnq_mig = zeros(TT, nalt)
		lnq_lft = zeros(TT, nalt)
		dlnq = zeros(TT, nalt)
		zbr = zeros(TT, nsim)

		@fastmath @inbounds for i in get_thread_range(nind)
			ind_sel = (1 + nalt * (i - 1)):(i * nalt)
			sim_sel = (1 + nsim * (i - 1)):(i * nsim)
			g = view(dgvec, i)
			ht = htvec[i]
			dage9 = dage9vec[i]

			individual_mnts!(mntvec_thread, mnt_range, mktshare, lftshare, lftpr_is, migpr_is,
							 locpr_is, lnqrnd_lftj, lnqrnd_migj, nalt_tmp, xf_xt_p, # <-- containers
							 xbm, ln1mlam, xbq, dlnq, lnq_mig, lnq_lft, zbr, #<-- containers again
							 bw, blft, bitr, bt, bl, bm, bf, bq, bqxj, bz, sigu, rhoq, sigq, #<-- endogeneous params
							 alpha, view(Delta, :, g), pr_lft[ht], view(pr_lft_alt, :, ht),
							 view(lnW, ind_sel), view(lnP, ind_sel), view(XT, :, i),
							 view(XL, :, i), view(XM, :, ind_sel), view(XF, :, ind_sel),
							 view(XQJ_mig, ind_sel), view(XQJ_lft, ind_sel),
							 view(XQ, :, ind_sel), dage9, view(ZSHK, :, sim_sel),
							 view(USHK, sim_sel), view(QSHK, sim_sel), nalt, nsim, unit)
			BLAS.axpy!(wgt[i], mntvec_thread, view(mntmat_thread, :, ht))
		end
		view(mntmat_all, :, :, Threads.threadid()) .= mntmat_thread
	end

	sum!(mntmat, mntmat_all)
	under9 = setdiff(1:mnt_len, mnt_cage9)
	broadcast!(/, view(mntmat, under9, :), view(mntmat, under9, :), sgwgt')

	# NOTE: remove constants in XF, XF_lft, XL_lft, and XT_lft
	mnt_par = vec(mean(view(mntmat, setdiff(1:mnt_len, [mnt_drop; mnt_cage9]), :),
					  weights(sgwgt), dims = 2))
	mnt_par_cage9 = view(mntmat, mnt_cage9, 2) / swgt9

	append!(mnt_par, mnt_par_cage9)
	prepend!(mnt_par, vec(mntmat[1:nalt, :]))
	return mnt_par
end


function get_moments(parm, alpha::T, lnW::AbstractVector{T},
					 lnP::AbstractVector{T}, XQJ_mig::AbstractVector{T},
					 XQJ_lft::AbstractVector{T},
					 XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					 XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					 XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
					 USHK::AbstractVector{T}, QSHK::AbstractVector{T},
					 pr_lft::AbstractVector{T}, pr_lft_alt::AbstractMatrix{T},
					 Delta::AbstractMatrix{T}, dgvec::AbstractVector{Int},
					 htvec::AbstractVector{Int}, dage9vec::AbstractVector{T},
					 wgt::AbstractVector{T}, sgwgt::AbstractVector{T},
					 swgt9::T, nind::Int, nalt::Int,
					 nsim::Int; xdim::Int = 1) where T <: AbstractFloat

	bw, blft, bitr, bqxj, bt, bl, bm, bf, bq, bz, sigu,
	rhoq, sigq, mnt_idx, mnt_drop, mnt_cage9 =
			unpack_parm_msm(parm, XT, XL, XM, XF, XQ, ZSHK, nalt, xdim)

	TT = promote_type(eltype(parm), eltype(lnW))
	unit = one(TT)
	mnt_len = sum(mnt_idx)
	mnt_range = get_mnt_range(mnt_idx)
	mntvec = zeros(TT, mnt_len)
	mntmat = zeros(TT, mnt_len, length(unique(htvec)))

	# --- setup containers ---
	mktshare = zeros(TT, nalt)
	lftshare = zeros(TT, nalt)
	lftpr_is = zeros(TT, nalt)
	migpr_is = zeros(TT, nalt)
	locpr_is = zeros(TT, nalt)
	lnqrnd_lftj = zeros(TT, nalt)
	lnqrnd_migj = zeros(TT, nalt)
	nalt_tmp = zeros(TT, nalt)
	xf_xt_p = zeros(TT, length(bf), length(bt))

	xbm = zeros(TT, nalt)
	xbq = zeros(TT, nalt)
	ln1mlam = zeros(TT, nalt)
	lnq_mig = zeros(TT, nalt)
	lnq_lft = zeros(TT, nalt)
	dlnq = zeros(TT, nalt)
	zbr = zeros(TT, nsim)

	@fastmath @inbounds @simd for i = 1:nind
		ind_sel = (1 + nalt * (i - 1)):(i * nalt)
		sim_sel = (1 + nsim * (i - 1)):(i * nsim)
		g = view(dgvec, i)
		ht = htvec[i]
		dage9 = dage9vec[i]

		individual_mnts!(mntvec, mnt_range, mktshare, lftshare, lftpr_is, migpr_is,
						 locpr_is, lnqrnd_lftj, lnqrnd_migj, nalt_tmp, xf_xt_p, # <-- containers
						 xbm, ln1mlam, xbq, dlnq, lnq_mig, lnq_lft, zbr, #<-- containers again
						 bw, blft, bitr, bt, bl, bm, bf, bq, bqxj, bz, sigu, rhoq, sigq, #<-- endogeneous params
		  				 alpha, view(Delta, :, g), pr_lft[ht], view(pr_lft_alt, :, ht),
						 view(lnW, ind_sel), view(lnP, ind_sel), view(XT, :, i),
						 view(XL, :, i), view(XM, :, ind_sel), view(XF, :, ind_sel),
						 view(XQJ_mig, ind_sel), view(XQJ_lft, ind_sel), view(XQ, :, ind_sel),
						 dage9, view(ZSHK, :, sim_sel), view(USHK, sim_sel),
						 view(QSHK, sim_sel), nalt, nsim, unit)
		BLAS.axpy!(wgt[i], mntvec, view(mntmat, :, ht))
	end

	under9 = setdiff(1:mnt_len, mnt_cage9)
	broadcast!(/, view(mntmat, under9, :), view(mntmat, under9, :), sgwgt')

	# NOTE: remove constants in XF, XF_lft, XL_lft, and XT_lft
	mnt_par = vec(mean(view(mntmat, setdiff(1:mnt_len, [mnt_drop; mnt_cage9]), :),
					  weights(sgwgt), dims = 2))
	mnt_par_cage9 = view(mntmat, mnt_cage9, 2) / swgt9

	append!(mnt_par, mnt_par_cage9)
	prepend!(mnt_par, vec(mntmat[1:nalt, :]))
	return mnt_par
end

#
# NOTE:
#	1. moments do NOT include location choice probabilities,
#	   this part of moments help to identify deltas, which
#	   is calculated by BLP contraction mapping.
# 	2. moments for lnQ SHOULD BE adjusted for age 9 and above only
#   3. lnW is de-meaned by average wage in age 9
#	4. dont't miss moments for lnW
#
using StatsFuns:log1pexp, logistic
using LinearAlgebra:dot
function individual_mnts!(mntvec, mnt_range, mktshare, lftshare, lftpr_is, migpr_is,
						  locpr_is, lnqrnd_lftj, lnqrnd_migj, nalt_tmp, xf_xt_p, # <-- containers
						  xbm, ln1mlam, xbq, dlnq, lnq_mig, lnq_lft, zbr, #<-- containers again
						  bw, blft, bitr, bt, bl, bm, bf, bq, bqxj, bz, sigu, rhoq, sigq, #<-- endogeneous params
  						  alpha, delta,	pr_lft_h, pr_lft_alt_h, lnw, lnp, xt, xl, xm, xf,  #<-- exogeneous params and data
						  qxj_mig, qxj_lft, xq, dage9, zshk, ushk, qshk, nalt, nsim, unit)
	##
	## mntvec = nalt + nXM + 2*nXF + 2*lnW + nXL + nXT + 2*nZS
	##		  + nZS * (lnW + ZQj) + 2*(nXQ + QJ + lnW) + 2
	##

	##
	## delta: 		J x 1 Vector
	## lnw, lnp: 	J x 1 Vector
	## qxj_mig, 	J x 1 Vector
	## qxj_lft,		J x 1 Vector
	## xbt, xbl: 	scalar
	## xbm: 		J x 1 Vector
	## ln1mlam, 	J x 1 Vector
	## xbq, 		scalar or J Vector
	## zbr, 		S x 1 Vector
	## zshk, 		nZ x S Matrix
	## ushk,		S x 1 Vector
	## qshk,		S x 1 Vector
	##

	nxf, nxt = size(xf_xt_p)

	# --- intermediate vars ---
	xbt = xt' * bt
	xbl = xl' * bl
	mul!(xbm, xm', bm)
	mul!(ln1mlam, xf', bf)
	broadcast!(nlog1pexp, ln1mlam, ln1mlam)

	mul!(xbq, xq', bq)
	lnq_alt!(lnq_mig, dlnq, lnw, ln1mlam, xbq, qxj_mig, qxj_lft, bw, blft, bitr, bqxj)
	broadcast!(+, lnq_lft, lnq_mig, dlnq)
	mul!(zbr, zshk', bz)

	# --- output containers ---
	TT = promote_type(eltype(bw), eltype(lnw))
	uc_lft_pr = zero(TT)
	fill!(mntvec, zero(TT))
	fill!(mktshare, zero(TT))
	fill!(lftshare, zero(TT))
	fill!(lnqrnd_lftj, zero(TT))
	fill!(lnqrnd_migj, zero(TT))
	# fill!(zm_mnt_lft, zero(TT))
	# fill!(zm_mnt_mig, zero(TT))
	# fill!(zlnw_mnt, zero(TT))
	# fill!(zlnqx_mnt, zero(TT))

    for s = 1:nsim
		zrnd = zbr[s]
        urnd = ushk[s]
        qrnd = qshk[s]
		theta = logistic(xbt + zrnd + sigu * urnd)
		lnqrnd = rhoq * sigq * urnd / sigu + sigq * sqrt(unit - rhoq^2) * qrnd

        for j = 1:nalt
            # --- 1. leftbh prob ---
			lftpr_is[j] = leftbh_prob(theta, ln1mlam[j], xbl, dlnq[j])
			migpr_is[j] = unit - lftpr_is[j]

            # --- 2. location specific utility ---
			gambar = gamfun(lnw[j], dlnq[j], lnq_mig[j], xbl, ln1mlam[j], theta)
            locpr_is[j] = Vloc(alpha, lnp[j], theta, xbm[j], gambar, delta[j])

        end # <-- end of j loop
		emaxprob!(locpr_is)

		ucpr_lft_is = dot(lftpr_is, locpr_is) # <-- unconditional left-behind prob.

		# --- 4-2. choice probability weighted lnqrnd: E(lnqrnd|k,j) ---
		BLAS.axpy!(lnqrnd, lftpr_is, lnqrnd_lftj)
		BLAS.axpy!(lnqrnd, migpr_is, lnqrnd_migj)

		# --- 5-1 moments for zshk: E(z|k) ---
		# BLAS.axpy!(ucpr_lft_is, zshk, zm_mnt_lft)
		# BLAS.axpy!(unit - ucpr_lft_is, zshk, zm_mnt_mig)
		BLAS.axpy!(ucpr_lft_is, view(zshk, :, s), view(mntvec, mnt_range[12]))
		BLAS.axpy!(unit - ucpr_lft_is, view(zshk, :, s), view(mntvec, mnt_range[11]))

		# --- 5-2 interaction between z and lnw: E(z'lnw|k) ---
		# broadcast!(*, nalt_tmp, lnw, lftpr_is, locpr_is)
		# BLAS.axpy!(sum(nalt_tmp), zshk, zlnw_mnt

		# unconditional covariance: E(z'lnw)
		# BLAS.axpy!(dot(lnw, locpr_is), zshk, zlnw_mnt)
		BLAS.axpy!(dot(lnw, locpr_is), view(zshk, :, s), view(mntvec, mnt_range[13]))

		# --- 5-3 interaction between z and qxj_lft, qxj_mig: E(z'qxj|k) ---
		# broadcast!(*, nalt_tmp, lnqx, lftpr_is, locpr_is)
		# BLAS.axpy!(sum(nalt_tmp), zshk, zlnqx_mnt)
		broadcast!((x, y) -> x - x * y, nalt_tmp, lftpr_is, locpr_is)
		BLAS.axpy!(dot(qxj_mig, nalt_tmp), view(zshk, :, s), view(mntvec, mnt_range[14]))

		broadcast!(*, nalt_tmp, lftpr_is, locpr_is)
		BLAS.axpy!(dot(qxj_lft, nalt_tmp), view(zshk, :, s), view(mntvec, mnt_range[15]))

		uc_lft_pr += ucpr_lft_is
        lftshare .+= lftpr_is
        mktshare .+= locpr_is

    end # <-- end of s loop

	# zm_mnt_lft ./= (nsim * pr_lft)
	# zm_mnt_mig ./= (nsim * (unit - pr_lft))
	broadcast!(/, view(mntvec, mnt_range[12]), view(mntvec, mnt_range[12]), nsim * pr_lft_h)
	broadcast!(/, view(mntvec, mnt_range[11]), view(mntvec, mnt_range[11]), nsim * (unit - pr_lft_h))
	# zlnw_mnt ./= nsim
	# zlnqx_mnt ./= nsim
	broadcast!(/, view(mntvec, mnt_range[13]), view(mntvec, mnt_range[13]), nsim)
	broadcast!(/, view(mntvec, mnt_range[15]), view(mntvec, mnt_range[15]), nsim * pr_lft_h)
	broadcast!(/, view(mntvec, mnt_range[14]), view(mntvec, mnt_range[14]), nsim * (unit - pr_lft_h))

	lftshare ./= nsim
	mktshare ./= nsim
	lnqrnd_lftj ./= nsim
	lnqrnd_migj ./= nsim

	copyto!(view(mntvec, mnt_range[1]), lftshare)
	uc_lft_pr /= nsim

	# --- moments for XM: E(xm) ---
	mul!(view(mntvec, mnt_range[2]), xm, mktshare)
	# mul!(xm_mnt, xm, mktshare)

	# --- moments for housing price lnP: E(lnp) ---
	# NOTE: NO need, already be contained in XF
	# view(mntvec, mnt_range[3]) .= dot(lnp, mktshare)
	# mlnp = dot(lnp, mktshare)

	# --- moments for XF (lambda): E(xf) & E(xf|k) ---
	# mul!(xf_mnt, xf, mktshare)
	mul!(view(mntvec, mnt_range[3]), xf, mktshare)
	broadcast!(*, nalt_tmp, lftshare, mktshare)
	# mul!(xf_mnt_lft, xf, nalt_tmp)
	mul!(view(mntvec, mnt_range[4]), xf, nalt_tmp)
	# xf_mnt_lft ./= pr_lft
	broadcast!(/, view(mntvec, mnt_range[4]), view(mntvec, mnt_range[4]), pr_lft_h)

	# --- moments for income lnW: E(lnw) & E(lnw|k) ---
	# mlnw = dot(lnw, mktshare)
	# mlnw_lft = dot(lnw, nalt_tmp) / pr_lft
	view(mntvec, mnt_range[5]) .= dot(lnw, mktshare)
	view(mntvec, mnt_range[6]) .= dot(lnw, nalt_tmp) / pr_lft_h

	# --- moments for XL: E(xl|k=1) ---
	# xl_mnt .= (uc_lft_pr / pr_lft) * xl
	BLAS.axpy!((uc_lft_pr / pr_lft_h), xl, view(mntvec, mnt_range[7])) #<-- NOTE: no constant in XL

	# --- moments for XT: E(xt|k=1) ---
	# xt_mnt .= (uc_lft_pr / pr_lft) * xt
	BLAS.axpy!((uc_lft_pr / pr_lft_h), xt, view(mntvec, mnt_range[8]))

	# ---  moments for XT: E(xt'lnw) and E(xf xt') ---
	# xt_lnw_mnt .= lnw_locpr * xt
	BLAS.axpy!(dot(lnw, mktshare), view(xt, 2:nxt), view(mntvec, mnt_range[9])) #<-- NOTE: drop constant in xt
	mul!(xf_xt_p, view(mntvec, mnt_range[3]), xt')
	view(mntvec, mnt_range[10]) .= vec(view(xf_xt_p, 2:nxf, 2:nxt)) #<-- NOTE: drop constant

	# --- E(xq'lnq | k = 0), E(lnw lnq | k = 0) ---
	broadcast!((x, y, z, u) -> x + y / (u - z), lnq_mig, lnq_mig, lnqrnd_migj, pr_lft_alt_h, unit)
	broadcast!(*, nalt_tmp, lnq_mig, mktshare, dage9)
	# mul!(xq_lnq_mig, xq, nalt_tmp)
	mul!(view(mntvec, mnt_range[16]), xq, nalt_tmp)
	# qxj_lnq_mig = dot(qxj_mig, nalt_tmp)
	view(mntvec, mnt_range[17]) .= dot(qxj_mig, nalt_tmp)
	# lnwq_mig = dot(lnw, nalt_tmp)
	view(mntvec, mnt_range[20]) .= dot(lnw, nalt_tmp)

	# --- E(lnq^2 | k = 0) ---
	# lnq2_mig = dot(lnq_mig, nalt_tmp)
	view(mntvec, mnt_range[22]) .= dot(lnq_mig, nalt_tmp)

	# --- E(xq'lnq | k = 1), E(lnw lnq | k = 1) ---
	broadcast!((x, y, z) -> x + y / z, lnq_lft, lnq_lft, lnqrnd_lftj, pr_lft_alt_h)
	broadcast!(*, nalt_tmp, lnq_lft, mktshare, dage9)
	# mul!(xq_lnq_lft, xq, nalt_tmp)
	mul!(view(mntvec, mnt_range[18]), xq, nalt_tmp)
	# qxj_lnq_lft = dot(qxj_lft, nalt_tmp)
	view(mntvec, mnt_range[19]) .= dot(qxj_lft, nalt_tmp)
	# lnwq_lft = dot(lnw, nalt_tmp)
	view(mntvec, mnt_range[21]) .= dot(lnw, nalt_tmp)

	# --- E(lnq^2 | k = 1) ---
	# lnq2_lft = dot(lnq_lft, nalt_tmp)
	view(mntvec, mnt_range[23]) .= dot(lnq_lft, nalt_tmp)

	# vcat(lftshare, xm_mnt, xf_mnt, xf_mnt_lft, mlnw, mlnw_lft, xl_mnt, xt_mnt,
	# 	   xt_lnw_mnt, xf_xt_p, zm_mnt_mig, zm_mnt_lft, zlnw_mnt, zqxj_mig_mnt,
	# 	   zqxj_lft_mnt, xq_lnq_mig, qxj_lnq_mig, xq_lnq_lft, qxj_lnq_lft,
	#	   lnwq_mig, lnwq_lft, lnq2_mig, lnq2_lft)
end

function unpack_parm_msm(parm, XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					 	 XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					 	 XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
					 	 nalt::Int, xdim::Int) where T <: AbstractFloat
	 nxt = size(XT, xdim)
	 nxl = size(XL, xdim)
	 nxm = size(XM, xdim)
	 nxf = size(XF, xdim)
	 nxq = size(XQ, xdim)
	 nzr = size(ZSHK, xdim)

	 bt = parm[1:nxt]
	 bl = parm[(nxt + 1):(nxt + nxl)]
	 bm = parm[(nxt + nxl + 1):(nxt + nxl + nxm)]
	 bf = parm[(nxt + nxl + nxm + 1):(nxt + nxl + nxm + nxf)]

	 blft = parm[nxt + nxl + nxm + nxf + 1]
	 bw = parm[nxt + nxl + nxm + nxf + 2]
	 bitr = parm[nxt + nxl + nxm + nxf + 3]
	 bqxj = parm[nxt + nxl + nxm + nxf + 4]
	 bq = parm[(nxt + nxl + nxm + nxf + 5):(nxt + nxl + nxm + nxf + nxq + 4)]

	 bz = parm[(nxt + nxl + nxm + nxf + nxq + 5):(nxt + nxl + nxm + nxf + nxq + nzr + 4)] #<- observed household char.

	 sigu = exp(parm[nxt + nxl + nxm + nxf + nxq + nzr + 5])
	 rhoq = tanh(parm[nxt + nxl + nxm + nxf + nxq + nzr + 6])
	 sigq = exp(parm[nxt + nxl + nxm + nxf + nxq + nzr + 7])

	 mnt_idx = [nalt, nxm, nxf, nxf, 1, 1, nxl, nxt, nxt - 1, (nxf - 1) * (nxt - 1),
	 			nzr, nzr, nzr, nzr, nzr, nxq, 1, nxq, 1, 1, 1, 1, 1]
	 mnt_drop = [collect(1:nalt); nalt + nxm + 1; nalt + nxm + nxf + 1;
	 			 nalt + nxm + 2*nxf + 3 + nxl]
	 mnt_cage9 = collect((sum(mnt_idx) - 2*nxq - 5):sum(mnt_idx))

	 return (bw, blft, bitr, bqxj, bt, bl, bm, bf, bq, bz, sigu, rhoq, sigq,
	 		 mnt_idx, mnt_drop, mnt_cage9)
end

function get_mnt_range(mnt_idx::AbstractVector{T}) where T <: Int
	mnt_end = cumsum(mnt_idx)
	mnt_bgn = [1; mnt_end[1:end-1] .+ 1]
	mnt_range = [mnt_bgn[i]:mnt_end[i] for i = 1:length(mnt_idx)]

	return mnt_range
end

function get_thread_range(n)
    tid = Threads.threadid()
    nt = Threads.nthreads()
    d , r = divrem(n, nt)
    from = (tid - 1) * d + min(r, tid - 1) + 1
    to = from + d - 1 + (tid <= r ? 1 : 0)
    from:to
end
