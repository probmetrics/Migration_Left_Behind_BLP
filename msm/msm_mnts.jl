function get_moments_adn(bw, blft, bitr, sigu, sigq, rhoq, psi,
						 bt, bl, bm, bf, bq, bqc, bz, XT, XL, XM, XF, XQ, XQC,
						 lnW::SharedArray{T, 1}, QSHK::SharedArray{T, 1}, USHK::SharedArray{T, 1},
						 ZSHK::SharedArray{T, 2}, pr_lft::SharedArray{T, 1},
						 prj_qlft::SharedArray{T, 2}, sdelta::SharedArray{T, 2}, ngvec::SharedArray{Int, 1},
						 htype_index::Array{Array{Int, 1}, 1}, wgt::SharedArray{T, 1},
						 age9dum::SharedArray{T, 1}, nind::Int, nmnt::Int, nalt::Int, nsim::Int) where T <: AbstractFloat
	mnt_mat = SharedArray{eltype(bw)}(nmnt, nind)
    for (ht, tg) in enumerate(htype_index)
		pr_lft_ht = pr_lft[ht]
		prj_qlft_ht = SharedArray(prj_qlft[:, ht])
        @sync @distributed for i in tg
            g = locate_gidx(i, ngvec)
        	mnt_mat[:, i] .= get_moments_adind(bw, blft, bitr, sigu, sigq, rhoq,
							psi, bt, bl, bm, bf, bq, bqc, bz,
							XT, XL, XM, XF, XQ, XQC, lnW, QSHK, USHK, ZSHK,
							pr_lft_ht, prj_qlft_ht, view(sdelta, :, g),
							age9dum, nalt, nsim, i)
        end
    end
	# @everywhere GC.gc()

	# --- construct moment conditions ---
	# TT = promote_type(eltype(bw), eltype(sdelta))
	# mktshare = zeros(TT, size(sdelta))
	# b0 = [1; ngvec[1:end-1] .+ 1]
	# for g = 1:length(ngvec)
	# 	idx = b0[g]:ngvec[g]
	# 	mktshare[:, g] = mean(view(mnt_mat, 1:nalt, idx), weights(view(wgt, idx)), 2)
	# end
	# tidx = Int(length(ngvec)/2)
	# mkt_vec_mnt = reshape(mktshare[2:end, :], (nalt - 1) * tidx, 2)
	# mnt_vec_ctrl = mean(view(mnt_mat, (nalt+1):nmnt, htype_index[1]), weights(view(wgt, htype_index[1])), 2)
	# mnt_vec_treat = mean(view(mnt_mat, (nalt+1):nmnt, htype_index[2]), weights(view(wgt, htype_index[2])), 2)
	# # 76 * 3 + 77 + 2 * ((size(XT, 1) - 1) + (size(XL, 1) - 1) + (size(XF, 1) - 2))
	# return vcat(mkt_vec_mnt,  hcat(mnt_vec_ctrl, mnt_vec_treat))

	mnt_vec_ctrl = mean(view(mnt_mat, :, htype_index[1]), weights(view(wgt, htype_index[1])), 2)
	mnt_vec_treat = mean(view(mnt_mat, :, htype_index[2]), weights(view(wgt, htype_index[2])), 2)
	return hcat(mnt_vec_ctrl, mnt_vec_treat)
end

#
# mnt_len = nXM + (nalt * 2 + nalt * nXF * 2) + nXL + J * nXT
#		  + nZS * 2 + nZS * (lnW + ZQj) * 2 + (nXQ + lnW) * 2 + 2
#
# NOTE:
#	1. moments do NOT include location choice probabilities,
#	   this part of moments help to identify deltas, which
#	   is calculated by BLP contraction mapping.
# 	2. combine moments for constant and xvars in XF
#	3. dont's miss moments for lnW
#
using StatsFuns:log1pexp, logistic
using LinearAlgebra:dot
function individual_mnts!(mntvec, mktshare, lftshare, lftpr_is, migpr_is, locpr_is, lnqrnd_lftj,
						  lnqrnd_migj, zm_mnt_mig, zm_mnt_lft, zlnw_mnt, zlnqx_mnt, zlnqx_mnt, xm_mnt,
						  xf_lftpr, xl_mnt, xt_mnt, xf_mnt, xq_lnq, nalt_tmp, # <-- containers
						  xbm, ln1mlam, xbq, dlnq, lnq_mig, lnq_lft, zbr, #<-- containers again
						  bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu, rhoq, sigq, #<-- endogeneous params
  						  alpha, delta,	prlj, yl, lnw, lnp, xt, xl, xm, xf, xq, zshk, #<-- exogeneous params and data
						  ushk, qshk, nalt, nsim)
	##
	## mntvec = nalt + (nXM + nlnP) + 2*nXF + 2*lnW + nXL + nXT + 2*nZS
	##		  + nZS * (lnW + ZQj) + (nXQ + lnW) + 1
	##

	##
	## delta: 		J x 1 Vector
	## lnw, lnp: 	J x 1 Vector
	## ym, yl:		J x 1 Vector
	## xbt, xbl: 	scalar
	## xbm: 		J x 1 Vector
	## ln1mlam, 	J x 1 Vector
	## xbq, 		scalar or J Vector
	## zbr, 		S x 1 Vector
	## ushk,		S x 1 Vector
	## qshk,		S x 1 Vector
	##

	# --- intermediate vars ---
	xbt = xt' * bt
	xbl = xl' * bl
	mul!(xbm, xm', bm)
	mul!(ln1mlam, xf', bf)
	broadcast!(log1pexp, ln1mlam, ln1mlam)

	mul!(xbq, xq', bq)
	lnq_alt!(lnq_mig, dlnq, lnw, ln1mlam, xbq, bw, blft, bitr)
	broadcast!(+, lnq_lft, lnq_mig, dlnq)
	mul!(zbr, zshk', bz)

	# --- output containers ---
	TT = promote_type(eltype(bw), eltype(lnw))
    unit = one(TT)
	fill!(mktshare, zero(TT))
	fill!(lftshare, zero(TT))
	fill!(lnqrnd_lftj, zero(TT))
	fill!(zm_mnt_lft, zero(TT))
	fill!(zm_mnt_mig, zero(TT))
	fill!(zlnw_mnt, zero(TT))
	fill!(zlnqx_mnt, zero(TT))

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
		BLAS.axpy!(ucpr_lft_is, zshk, zm_mnt_lft)
		BLAS.axpy!(unit - ucpr_lft_is, zshk, zm_mnt_mig)

		# --- 5-2 interaction between z and lnw: E(z'lnw|k) ---
		# broadcast!(*, nalt_tmp, lnw, lftpr_is, locpr_is)
		# BLAS.axpy!(sum(nalt_tmp), zshk, zlnw_mnt

		# unconditional covariance: E(z'lnw)
		BLAS.axpy!(dot(lnw, locpr_is), zshk, zlnw_mnt)

		# --- 5-3 interaction between z and lnqx: E(z'lnqx|k) ---
		# broadcast!(*, nalt_tmp, lnqx, lftpr_is, locpr_is)
		# BLAS.axpy!(sum(nalt_tmp), zshk, zlnqx_mnt)

		# unconditional covariance: E(z'lnqx)
		BLAS.axpy!(dot(lnqx, locpr_is), zshk, zlnw_mnt)

        lftshare .+= lftpr_is
        mktshare .+= locpr_is

    end # <-- end of s loop

	zm_mnt_lft ./= nsim
	zm_mnt_mig ./= nsim
	zlnw_mnt ./= nsim
	zlnqx_mnt ./= nsim
	lftshare ./= nsim
	mktshare ./= nsim
	lnqrnd_lftj ./= nsim
	lnqrnd_migj ./= nsim

	uc_lft_pr = dot(lftshare, mktshare)

	# --- moments for XM: E(xm) ---
	mul!(xm_mnt, xm, mktshare)

	# --- moments for housing price lnP: E(lnp) ---
	mlnp = dot(lnp, mktshare)

	# --- moments for XF (lambda): E(xf) & E(xf|k) ---
	mul!(xf_mnt, xf, mktshare)
	broadcast!(*, xf_lftpr, xf, lftshare') * mktshare

	# --- moments for income lnW: E(lnw) & E(lnw|k) ---
	mlnw = dot(lnw, mktshare)
	broadcast!(*, nalt_tmp, lftshre, mktshare)
	mlnw_lft = dot(lnw, nalt_tmp)

	# --- moments for XL: E(xl|k) ---
	xl_mnt .= uc_lft_pr * xl

	# --- moments for XT: E(xt|k) ---
	xt_mnt .= uc_lft_pr * xt

	# --- E(xq'lnq | k = 1), E(lnw lnq | k = 1) ---
	broadcast!((x, y, z) -> x + y / z, lnq_lft, lnqrnd_lftj, prlj)
	broadcast!(*, nalt_tmp, lnq_lft, mktshare)
	mul!(xq_lnq_lft, xq, nalt_tmp)
	lnwq_lft = dot(lnw, nalt_tmp)

	# --- E(lnq^2 | k = 1) ---
	lnq2_lft = dot(lnq_lft, nalt_tmp)

	# --- E(xq'lnq | k = 0), E(lnw lnq | k = 0) ---
	broadcast!((x, y, z) -> x + y / (unit - z), lnq_mig, lnqrnd_migj, prlj)
	broadcast!(*, nalt_tmp, lnq_mig, mktshare)
	mul!(xq_lnq_mig, xq, nalt_tmp)
	lnwq_mig = dot(lnw, nalt_tmp)

	# # --- E(lnq^2 | k = 0) ---
	lnq2_mig = dot(lnq_mig, nalt_tmp)

	vcat(lftshare, xm_mnt, mlnp, xf_mnt, xf_lftpr, mlnw, mlnw_lft,, xl_mnt, xt_mnt,
		 zm_mnt_mig, zm_mnt_lft, zlnw_mnt, zlnqx_mnt, xq_lnq_lft, xq_lnq_mig, lnwq_lft,
		 lnwq_mig, lnq2_lft, lnq2_mig)
end
