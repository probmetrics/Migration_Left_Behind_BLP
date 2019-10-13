function msm_obj(parm, data_mnts::AbstractVector{T}, dwt::AbstractVector{T},
				 alpha::T, lnW::AbstractVector{T}, lnP::AbstractVector{T},
				 XQJ_mig::AbstractMatrix{T}, XT::AbstractMatrix{T},
				 XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
				 XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
				 ZSHK::AbstractMatrix{T},
				 QSHK::AbstractVector{T}, pr_lft::AbstractVector{T},
				 Delta, dgvec::AbstractVector{Int}, htvec::AbstractVector{Int},
				 dage9vec::AbstractVector{T}, wgt::AbstractVector{T},
				 sgwgt::AbstractVector{T}, shtwgt::AbstractVector{T},
				 swgt9::T, nind::Int, nalt::Int,
				 nsim::Int; xdim::Int = 1) where T <: AbstractFloat

	DT = promote_type(eltype(Delta), eltype(parm))
	# delta_old = deepcopy(Delta)
	delta_fpt = zeros(DT, size(Delta))
	delta_new = zeros(DT, size(Delta))
	delta_q1 = zeros(DT, size(Delta))
	delta_q2 = zeros(DT, size(Delta))

	fpt_squarem!(delta_fpt, delta_new, Delta, delta_q1, delta_q2, lnDataShare,
				 parm, lnW, lnP, XQJ_mig, XT, XL, XM, XF, XQ, ZSHK,
				 wgt, sgwgt, nind, nalt, nsim, dgvec; alpha = alpha,
				 xdim = xdim)

	mnt_par = get_moments_thread(parm, alpha, lnW, lnP, XQJ_mig, XT, XL, XM, XF, XQ,
								 ZSHK, QSHK, pr_lft, delta_fpt, dgvec, htvec,
								 dage9vec, wgt, shtwgt, swgt9, nind,
								 nalt, nsim; xdim = xdim)

	gmnt = mnt_par - data_mnts
	broadcast!(*, gmnt, gmnt, dwt, gmnt)
	obj = sum(gmnt)
 	return obj
end

function get_moments_thread(parm, alpha::T, lnW::AbstractVector{T},
					 lnP::AbstractVector{T}, XQJ_mig::AbstractMatrix{T},
					 XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					 XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					 XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
					 QSHK::AbstractVector{T}, pr_lft::AbstractVector{T},
					 Delta, dgvec::AbstractVector{Int},
					 htvec::AbstractVector{Int}, dage9vec::AbstractVector{T},
					 wgt::AbstractVector{T}, shtwgt::AbstractVector{T},
					 swgt9::T, nind::Int, nalt::Int,
					 nsim::Int; xdim::Int = 1) where T <: AbstractFloat

	bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz,
	sigq, mnt_idx, mnt_drop, mnt_cage9 =
			unpack_parm_msm(parm, XT, XL, XM, XF, XQ, XQJ_mig, ZSHK, nalt, xdim)

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
		locpr_is = zeros(TT, nalt)
		locmig_pr = zeros(TT, nalt)
		loclft_pr = zeros(TT, nalt)
		nalt_tmp = zeros(TT, nalt)
		xf_wmkt = zeros(TT, length(bf))
		xf_xt_p = zeros(TT, length(bf), length(bt))
		xt_xqj_p = zeros(TT, length(bt), length(bqj_mig))

		xbm = zeros(TT, nalt)
		xbqj_mig = zeros(TT, nalt)
		xbqj_dif = zeros(TT, nalt)
		ln1mlam = zeros(TT, nalt)
		lnq_mig = zeros(TT, nalt)
		lnq_lft = zeros(TT, nalt)
		dlnq = zeros(TT, nalt)
		zbr = zeros(TT, nsim)
		xqj_mig_ntmp = zeros(TT, length(bqj_mig))

		@fastmath @inbounds for i in get_thread_range(nind)
			ind_sel = (1 + nalt * (i - 1)):(i * nalt)
			sim_sel = (1 + nsim * (i - 1)):(i * nsim)
			g = dgvec[i]
			ht = htvec[i]
			dage9 = dage9vec[i]

			individual_mnts!(mntvec_thread, mnt_range, mktshare, lftshare, lftpr_is,
							 locpr_is, locmig_pr, loclft_pr, nalt_tmp, xf_wmkt,
							 xf_xt_p, xt_xqj_p, xbm, ln1mlam, # <-- containers
							 xbqj_mig, xbqj_dif, dlnq, lnq_mig, lnq_lft, zbr, xqj_mig_ntmp,  #<-- containers again
							 bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, sigq, #<-- endogeneous params
							 alpha, view(Delta, :, g), pr_lft[ht],
							 view(lnW, ind_sel), view(lnP, ind_sel), view(XT, :, i),
							 view(XL, :, i), view(XM, :, ind_sel), view(XF, :, ind_sel),
							 view(XQJ_mig, :, ind_sel), view(XQ, :, i), dage9, view(ZSHK, :, sim_sel),
							 view(QSHK, sim_sel), nalt, nsim, unit)
			BLAS.axpy!(wgt[i], mntvec_thread, view(mntmat_thread, :, ht))
		end
		view(mntmat_all, :, :, Threads.threadid()) .= mntmat_thread
	end

	sum!(mntmat, mntmat_all)
	under9 = setdiff(1:mnt_len, mnt_cage9)
	broadcast!(/, view(mntmat, under9, :), view(mntmat, under9, :), shtwgt')

	# NOTE: remove constants in XF, XF_lft, XL_lft, and XT_lft
	mnt_par = vec(mean(view(mntmat, setdiff(1:mnt_len, [mnt_drop; mnt_cage9]), :),
					  weights(shtwgt), dims = 2))
	mnt_par_cage9 = view(mntmat, mnt_cage9, 2) / swgt9

	append!(mnt_par, mnt_par_cage9)
	prepend!(mnt_par, vec(mntmat[1:nalt, :]))
	return mnt_par
end


function get_moments(parm, alpha::T, lnW::AbstractVector{T},
					 lnP::AbstractVector{T}, XQJ_mig::AbstractMatrix{T},
					 XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					 XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					 XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
					 QSHK::AbstractVector{T}, pr_lft::AbstractVector{T},
					 Delta, dgvec::AbstractVector{Int},
					 htvec::AbstractVector{Int}, dage9vec::AbstractVector{T},
					 wgt::AbstractVector{T}, shtwgt::AbstractVector{T},
					 swgt9::T, nind::Int, nalt::Int,
					 nsim::Int; xdim::Int = 1) where T <: AbstractFloat

	bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz,
	sigq, mnt_idx, mnt_drop, mnt_cage9 =
			unpack_parm_msm(parm, XT, XL, XM, XF, XQ, XQJ_mig, ZSHK, nalt, xdim)

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
	locpr_is = zeros(TT, nalt)
	locmig_pr = zeros(TT, nalt)
	loclft_pr = zeros(TT, nalt)
	nalt_tmp = zeros(TT, nalt)
	xf_wmkt = zeros(TT, length(bf))
	xf_xt_p = zeros(TT, length(bf), length(bt))
	xt_xqj_p = zeros(TT, length(bt), length(bqj_mig))

	xbm = zeros(TT, nalt)
	xbqj_mig = zeros(TT, nalt)
	xbqj_dif = zeros(TT, nalt)
	ln1mlam = zeros(TT, nalt)
	lnq_mig = zeros(TT, nalt)
	lnq_lft = zeros(TT, nalt)
	dlnq = zeros(TT, nalt)
	zbr = zeros(TT, nsim)
	xqj_mig_ntmp = zeros(TT, length(bqj_mig))

	@fastmath @inbounds @simd for i = 1:nind
		ind_sel = (1 + nalt * (i - 1)):(i * nalt)
		sim_sel = (1 + nsim * (i - 1)):(i * nsim)
		g = dgvec[i]
		ht = htvec[i]
		dage9 = dage9vec[i]

		individual_mnts!(mntvec, mnt_range, mktshare, lftshare, lftpr_is,
						 locpr_is, locmig_pr, loclft_pr, nalt_tmp, xf_wmkt,
						 xf_xt_p, xt_xqj_p, xbm, ln1mlam, xbqj_mig, # <-- containers
						 xbqj_dif, dlnq, lnq_mig, lnq_lft, zbr, xqj_mig_ntmp, #<-- containers again
						 bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, sigq, #<-- endogeneous params
		  				 alpha, view(Delta, :, g), pr_lft[ht],
						 view(lnW, ind_sel), view(lnP, ind_sel), view(XT, :, i),
						 view(XL, :, i), view(XM, :, ind_sel), view(XF, :, ind_sel),
						 view(XQJ_mig, :, ind_sel), view(XQ, :, i),
						 dage9, view(ZSHK, :, sim_sel),
						 view(QSHK, sim_sel), nalt, nsim, unit)
		BLAS.axpy!(wgt[i], mntvec, view(mntmat, :, ht))
	end

	under9 = setdiff(1:mnt_len, mnt_cage9)
	broadcast!(/, view(mntmat, under9, :), view(mntmat, under9, :), shtwgt')

	# NOTE: remove constants in XF_mig, XF_lft, XL_lft, and XT_lft
	mnt_par = vec(mean(view(mntmat, setdiff(1:mnt_len, [mnt_drop; mnt_cage9]), :),
					  weights(shtwgt), dims = 2))
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
function individual_mnts!(mntvec, mnt_range, mktshare, lftshare, lftpr_is,
						  locpr_is, locmig_pr, loclft_pr, nalt_tmp, xf_wmkt,
						  xf_xt_p, xt_xqj_p, xbm, ln1mlam, xbqj_mig,  # <-- containers
						  xbqj_dif, dlnq, lnq_mig, lnq_lft, zbr, xqj_mig_ntmp, #<-- containers again
						  bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, sigq, #<-- endogeneous params
  						  alpha, delta,	pr_lft_h, lnw, lnp, xt, xl, xm, xf,  #<-- exogeneous params and data
						  xqj_mig, xq, dage9, zshk, qshk, nalt, nsim, unit)
	##
	## mntvec = nalt + nXM + 2*nXF + 2*lnW + nXL + nXT + 2*nZS
	##		  + nZS * (lnW + ZQj) + 2*(nXQ + QJ + lnW) + 2
	##

	##
	## delta: 		J x 1 Vector
	## lnw, lnp: 	J x 1 Vector
	## xqj_mig, 	nxqj x J Matrix
	## xbt, xbl: 	scalar
	## xbm: 		J x 1 Vector
	## ln1mlam, 	J x 1 Vector
	## xbq, 		scalar
	## zbr, 		S x 1 Vector
	## zshk, 		nZ x S Matrix
	## ushk,		S x 1 Vector
	## qshk,		S x 1 Vector
	##

	nxf, nxt = size(xf_xt_p)

	# --- intermediate vars ---
	pr_mig_h = unit - pr_lft_h
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

	# --- output containers ---
	TT = promote_type(eltype(bw), eltype(lnw))
	uc_lft_pr = zero(TT)
	lnqrnd_lft = zero(TT)
	lnqrnd_mig = zero(TT)
	lnqrnd2_lft = zero(TT)
	lnqrnd2_mig = zero(TT)
	lnw_qrnd_lft = zero(TT)
	lnw_qrnd_mig = zero(TT)

	fill!(mntvec, zero(TT))
	fill!(mktshare, zero(TT))
	fill!(lftshare, zero(TT))
	fill!(locmig_pr, zero(TT))
	fill!(loclft_pr, zero(TT))

    for s = 1:nsim
		zrnd = zbr[s]
        qrnd = qshk[s]
		theta = logistic(xbt + zrnd)
		# lnqrnd = rhoq * sigq * urnd / sigu + sigq * sqrt(unit - rhoq^2) * qrnd
		lnqrnd = sigq * qrnd

        for j = 1:nalt
            # --- 1. leftbh prob ---
			lftpr_is[j] = leftbh_prob(theta, ln1mlam[j], xbl, dlnq[j])

            # --- 2. location specific utility ---
			gambar = gamfun(lnw[j], dlnq[j], lnq_mig[j], xbl, ln1mlam[j], theta)
            locpr_is[j] = Vloc(alpha, lnp[j], theta, xbm[j], gambar, delta[j])

        end # <-- end of j loop
		emaxprob!(locpr_is)

		ucpr_lft_is = dot(lftpr_is, locpr_is) # <-- unconditional left-behind prob.

		# --- 4-2. choice probability weighted lnqrnd: E(lnqrnd|k) ---
		lnqrnd_lft += lnqrnd * ucpr_lft_is
		lnqrnd_mig += lnqrnd * (unit - ucpr_lft_is)
		lnqrnd2_lft += lnqrnd^2 * ucpr_lft_is
		lnqrnd2_mig += lnqrnd^2 * (unit - ucpr_lft_is)

		# --- 5-1 moments for zshk: E(z|k) ---
		BLAS.axpy!(ucpr_lft_is, view(zshk, :, s), view(mntvec, mnt_range[12]))
		BLAS.axpy!(unit - ucpr_lft_is, view(zshk, :, s), view(mntvec, mnt_range[11]))

		# --- 5-2 interaction between z and lnw: E(z'lnw) ---
		BLAS.axpy!(dot(lnw, locpr_is), view(zshk, :, s), view(mntvec, mnt_range[13]))

		## NOTE important intermediate var
		broadcast!((x, y) -> x - x * y, nalt_tmp, locpr_is, lftpr_is)

		lnw_qrnd_mig += dot(lnw, nalt_tmp) * lnqrnd # <-- E(lnw*uq|k=0)

		# --- 5-3 interaction between z and xqj_mig: E(z'xqj|k) ---
		mul!(xqj_mig_ntmp, xqj_mig, nalt_tmp)
		# mul!(zshk_xqj, view(zshk, :, s), xqj_mig_ntmp')
		# BLAS.axpy!(unit, vec(zshk_xqj), view(mntvec, mnt_range[14]))
		outer_vec!(view(mntvec, mnt_range[14]), view(zshk, :, s), xqj_mig_ntmp)
		BLAS.axpy!(lnqrnd, xqj_mig_ntmp, view(mntvec, mnt_range[17])) # <-- E(xqj*uq|k=0)

		locmig_pr .+= nalt_tmp

		## NOTE important intermediate var
		broadcast!(*, nalt_tmp, lftpr_is, locpr_is)

		lnw_qrnd_lft += dot(lnw, nalt_tmp) * lnqrnd # <-- E(lnw*uq|k=1)

		mul!(xqj_mig_ntmp, xqj_mig, nalt_tmp)
		# mul!(zshk_xqj, view(zshk, :, s), xqj_mig_ntmp')
		# BLAS.axpy!(unit, vec(zshk_xqj), view(mntvec, mnt_range[15]))
		outer_vec!(view(mntvec, mnt_range[15]), view(zshk, :, s), xqj_mig_ntmp)

		BLAS.axpy!(lnqrnd, xqj_mig_ntmp, view(mntvec, mnt_range[19])) # <-- E(xqj*uq|k=1)

		loclft_pr .+= nalt_tmp

		uc_lft_pr += ucpr_lft_is
        lftshare .+= lftpr_is
        mktshare .+= locpr_is

    end # <-- end of s loop

	# zm_mnt_lft ./= (nsim * pr_lft)
	# zm_mnt_mig ./= (nsim * (unit - pr_lft))
	broadcast!(/, view(mntvec, mnt_range[12]), view(mntvec, mnt_range[12]), nsim * pr_lft_h)
	broadcast!(/, view(mntvec, mnt_range[11]), view(mntvec, mnt_range[11]), nsim * pr_mig_h)
	# zlnw_mnt ./= nsim
	# zlnxqj_mnt ./= nsim
	broadcast!(/, view(mntvec, mnt_range[13]), view(mntvec, mnt_range[13]), nsim)
	broadcast!(/, view(mntvec, mnt_range[15]), view(mntvec, mnt_range[15]), nsim * pr_lft_h)
	broadcast!(/, view(mntvec, mnt_range[14]), view(mntvec, mnt_range[14]), nsim * pr_mig_h)
	# E(xqj*uq|k)
	broadcast!(*, view(mntvec, mnt_range[17]), view(mntvec, mnt_range[17]), dage9 / (nsim * pr_mig_h))
	broadcast!(*, view(mntvec, mnt_range[19]), view(mntvec, mnt_range[19]), dage9 / (nsim * pr_lft_h))

	lftshare ./= nsim
	mktshare ./= nsim
	locmig_pr ./= (nsim * pr_mig_h)
	loclft_pr ./= (nsim * pr_lft_h)

	lnqrnd_lft /= (nsim * pr_lft_h)
	lnqrnd_mig /= (nsim * pr_mig_h)
	lnqrnd2_lft /= (nsim * pr_lft_h)
	lnqrnd2_mig /= (nsim * pr_mig_h)
	lnw_qrnd_lft /= (nsim * pr_lft_h)
	lnw_qrnd_mig /= (nsim * pr_mig_h)

	copyto!(view(mntvec, mnt_range[1]), lftshare)
	uc_lft_pr /= nsim

	# --- moments for XM: E(xm) ---
	mul!(view(mntvec, mnt_range[2]), xm, mktshare)
	# mul!(xm_mnt, xm, mktshare)

	# --- moments for housing price lnP: E(lnp) ---
	# NOTE: NO need, already be contained in XF
	# view(mntvec, mnt_range[3]) .= dot(lnp, mktshare)
	# mlnp = dot(lnp, mktshare)

	# --- moments for XFW (lambda): E(xfw | k = 0) & E(xfw|k = 1) ---
	mul!(view(mntvec, mnt_range[3]), xf, locmig_pr)
	mul!(view(mntvec, mnt_range[4]), xf, loclft_pr)

	# --- moments for income lnW: E(lnw | k = 0) & E(lnw|k = 1) ---
	# view(mntvec, mnt_range[5]) .= dot(lnw, locmig_pr)
	# view(mntvec, mnt_range[6]) .= dot(lnw, loclft_pr)

	# --- moments for XL: E(xl|k=1) ---
	# xl_mnt .= (uc_lft_pr / pr_lft) * xl
	BLAS.axpy!((uc_lft_pr / pr_lft_h), xl, view(mntvec, mnt_range[5])) 

	# --- moments for XT: E(xt|k=1) ---
	# xt_mnt .= (uc_lft_pr / pr_lft) * xt
	BLAS.axpy!((uc_lft_pr / pr_lft_h), xt, view(mntvec, mnt_range[6]))

	# ---  moments for XT: E(xt'lnw) and E(xf xt' | k = 1) ---
	# xt_lnw_mnt .= lnw_locpr * xt
	BLAS.axpy!(dot(lnw, mktshare), xt, view(mntvec, mnt_range[7]))

	mul!(xf_wmkt, xf, mktshare)
	mul!(xf_xt_p, xf_wmkt, xt')
	view(mntvec, mnt_range[8]) .= vec(view(xf_xt_p, 2:nxf, :)) #<-- NOTE: drop constant

	# ---  moments for XT: E(xt'xqj_mig) and E(xt'xqj_lft) ---
	mul!(xqj_mig_ntmp, xqj_mig, locmig_pr)
	mul!(xt_xqj_p, xt, xqj_mig_ntmp')
	view(mntvec, mnt_range[9]) .= vec(xt_xqj_p)

	mul!(xqj_mig_ntmp, xqj_mig, loclft_pr)
	mul!(xt_xqj_p, xt, xqj_mig_ntmp')
	view(mntvec, mnt_range[10]) .= vec(xt_xqj_p)

	# --- E(xq'lnq | k = 0) ---
	lnq_mig_i = dot(lnq_mig, mktshare) + lnqrnd_mig
	BLAS.axpy!(dage9*lnq_mig_i, xq, view(mntvec, mnt_range[16]))

	# --- E(xqj'lnq | k = 0) ---
	broadcast!(*, nalt_tmp, lnq_mig, mktshare)

	mul!(xqj_mig_ntmp, xqj_mig, nalt_tmp)
	BLAS.axpy!(dage9, xqj_mig_ntmp, view(mntvec, mnt_range[17]))
	# view(mntvec, mnt_range[17]) .= dot(xqj_mig, nalt_tmp)

	# --- E(lnw lnq | k = 0) ---
	view(mntvec, mnt_range[20]) .= dage9 * (dot(lnw, nalt_tmp) + lnw_qrnd_mig)

	# --- E(lnq^2 | k = 0) ---
	# lnq2_mig = dot(lnq_mig, nalt_tmp)
	view(mntvec, mnt_range[22]) .= dage9 * (dot(lnq_mig, nalt_tmp) + lnqrnd2_mig)

	# --- E(xq'lnq | k = 1) ---
	lnq_lft_i = dot(lnq_lft, mktshare) + lnqrnd_lft
	BLAS.axpy!(dage9 * lnq_lft_i, xq, view(mntvec, mnt_range[18]))

	# --- E(xqj'lnq | k = 1) ---
	broadcast!(*, nalt_tmp, lnq_lft, mktshare)

	mul!(xqj_mig_ntmp, xqj_mig, nalt_tmp)
	BLAS.axpy!(dage9, xqj_mig_ntmp, view(mntvec, mnt_range[19]))
	# view(mntvec, mnt_range[19]) .= dot(xqj_lft, nalt_tmp)

	# --- E(lnw lnq | k = 1) ---
	view(mntvec, mnt_range[21]) .= dage9 * (dot(lnw, nalt_tmp) + lnw_qrnd_lft)

	# --- E(lnq^2 | k = 1) ---
	# lnq2_lft = dot(lnq_lft, nalt_tmp)
	view(mntvec, mnt_range[23]) .= dage9 * (dot(lnq_lft, nalt_tmp) + lnqrnd2_lft)

	#
	# vcat(lftshare(154), xm_mnt(15), xfw_mnt_mig(7), xfw_mnt_lft(7), 
	# 	   xl_mnt_lft(4), xt_mnt_lft(5), xt_lnw_mnt(5),
	#
	# 	   xf_xt(30), xt_xqj_mig(10), xt_xqj_lft(10)
	#
	# 	   zm_mnt_mig(2), zm_mnt_lft(2),
	# 	   zlnw_mnt(2), zxqj_mig_mnt(4), zxqj_lft_mnt(4), xq_lnq_mig(6), xqj_lnq_mig(2),
	# 	   xq_lnq_lft(6), xqj_lnq_lft(2), lnwq_mig(1), lnwq_lft(1), lnq2_mig(1), lnq2_lft(1))
	#
end

function unpack_parm_msm(parm, XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					 	 XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					 	 XQ::AbstractMatrix{T}, XQJ_mig::AbstractMatrix{T},
						 ZSHK::AbstractMatrix{T}, nalt::Int, xdim::Int) where T <: AbstractFloat
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
	 bqj_mig = parm[(nxt + nxl + nxm + nxf + nxq + 4):(nxt + nxl + nxm + nxf + nxq + 3 + nxqj)]
	 bqj_dif = parm[(nxt + nxl + nxm + nxf + nxq + 4 + nxqj):(nxt + nxl + nxm + nxf + nxq + 3 + 2*nxqj)]

	 bz = parm[(nxt + nxl + nxm + nxf + nxq + 2*nxqj + 4):(nxt + nxl + nxm + nxf + nxq + 2*nxqj + nzr + 3)] #<- observed household char.
	 sigq = exp(parm[nxt + nxl + nxm + nxf + nxq + 2*nxqj + nzr + 4])

	 mnt_idx = [nalt, nxm, nxf, nxf, nxl, nxt, nxt, (nxf - 1) * nxt,
	 			nxt * nxqj, nxt * nxqj, nzr, nzr, nzr, 
	 			nzr*nxqj, nzr*nxqj, nxq, nxqj, nxq, nxqj, 1, 1, 1, 1]

	 loctmp = nalt + nxm + 2*nxf + nxl + nxt
	 locskip = (nxf - 1) * nxt + nxt * nxqj * 2
	 mnt_drop = [1:nalt; nalt + nxm + 1; nalt + nxm + nxf + 1;
	 			 nalt + nxm + 2*nxf + 1; (loctmp + 1):(loctmp + nxt + locskip);
				 (loctmp + nxt + locskip + 2*nzr + 1):(loctmp + nxt + locskip + 3*nzr)]

	 mnt_cage9 = collect((sum(mnt_idx) - 2*nxq - 2*nxqj - 3):sum(mnt_idx))

	 return (bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz, sigq,
	 		 mnt_idx, mnt_drop, mnt_cage9)
end

function unpack_parm(parm, XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
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
	 bqj_mig = parm[(nxt + nxl + nxm + nxf + nxq + 4):(nxt + nxl + nxm + nxf + nxq + 3 + nxqj)]
	 bqj_dif = parm[(nxt + nxl + nxm + nxf + nxq + 4 + nxqj):(nxt + nxl + nxm + nxf + nxq + 3 + 2*nxqj)]

	 bz = parm[(nxt + nxl + nxm + nxf + nxq + 2*nxqj + 4):(nxt + nxl + nxm + nxf + nxq + 2*nxqj + nzr + 3)] #<- observed household char.
	 return (bw, blft, bitr, bt, bl, bm, bf, bq, bqj_mig, bqj_dif, bz)
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


function outer_vec!(R, A, B)
	# length(R) == length(A) * length(B) || error("dimension mismatch between R and A*B")
	k = 1
	@fastmath @inbounds for i = eachindex(B)
		for j = eachindex(A)
			R[k] += A[j] * B[i]
			k += 1
		end
	end
end
