function get_moments_adn(bw, blft, bitr, sigu, sigq, rhoq, psi,
						 bt, bl, bm, bf, bq, bqc, bz, XT, XL, XM, XF, XQ, XQC,
						 lnW::SharedArray{T, 1}, QSHK::SharedArray{T, 1}, USHK::SharedArray{T, 1},
						 ZSHK::SharedArray{T, 2}, pr_lft::SharedArray{T, 1},
						 prj_qlft::SharedArray{T, 2}, sdelta::SharedArray{T, 2}, ngvec::SharedArray{Int, 1},
						 htype_index::Array{Array{Int, 1}, 1}, wgt::SharedArray{T, 1},
						 age9dum::SharedArray{T, 1}, nind::Int, nmnt::Int, nalt::Int, nsim::Int) where T <: AbstractFloat

	mnt_len = sum(mnt_idx)
	mnt_range = get_mnt_range(mnt_idx)
	mntvec = zeros(TT, mnt_len)


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
function individual_mnts!(mntvec, mnt_range, mktshare, lftshare, lftpr_is, migpr_is,
						  locpr_is, lnqrnd_lftj, lnqrnd_migj, nalt_tmp, # <-- containers
						  xbm, ln1mlam, xbq, dlnq, lnq_mig, lnq_lft, zbr, #<-- containers again
						  bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu, rhoq, sigq, #<-- endogeneous params
  						  alpha, delta,	pr_lft, pr_lft_alt, lnw, lnp, xt, xl, xm, xf,  #<-- exogeneous params and data
						  lnqx, xq, zshk, ushk, qshk, nalt, nsim)
	##
	## mntvec = nalt + (nXM + nlnP) + 2*nXF + 2*lnW + nXL + nXT + 2*nZS
	##		  + nZS * (lnW + ZQj) + 2*(nXQ + lnW) + 2
	##

	##
	## delta: 		J x 1 Vector
	## lnw, lnp: 	J x 1 Vector
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
	fill!(mntvec, zero(TT))
	fill!(mktshare, zero(TT))
	fill!(lftshare, zero(TT))
	fill!(lnqrnd_lftj, zero(TT))
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
		BLAS.axpy!(ucpr_lft_is, zshk, view(mntvec, mnt_range[11]))
		BLAS.axpy!(unit - ucpr_lft_is, zshk, view(mntvec, mnt_range[10]))

		# --- 5-2 interaction between z and lnw: E(z'lnw|k) ---
		# broadcast!(*, nalt_tmp, lnw, lftpr_is, locpr_is)
		# BLAS.axpy!(sum(nalt_tmp), zshk, zlnw_mnt

		# unconditional covariance: E(z'lnw)
		# BLAS.axpy!(dot(lnw, locpr_is), zshk, zlnw_mnt)
		BLAS.axpy!(dot(lnw, locpr_is), zshk, view(mntvec, mnt_range[12]))

		# --- 5-3 interaction between z and lnqx: E(z'lnqx|k) ---
		# broadcast!(*, nalt_tmp, lnqx, lftpr_is, locpr_is)
		# BLAS.axpy!(sum(nalt_tmp), zshk, zlnqx_mnt)

		# unconditional covariance: E(z'lnqx)
		# BLAS.axpy!(dot(lnqx, locpr_is), zshk, zlnqx_mnt)
		BLAS.axpy!(dot(lnqx, locpr_is), zshk, view(mntvec, mnt_range[13]))

        lftshare .+= lftpr_is
        mktshare .+= locpr_is

    end # <-- end of s loop

	# zm_mnt_lft ./= (nsim * pr_lft)
	# zm_mnt_mig ./= (nsim * (unit - pr_lft))
	view(mntvec, mnt_range[11]) ./= (nsim * pr_lft)
	view(mntvec, mnt_range[10]) ./= (nsim * (unit - pr_lft))
	# zlnw_mnt ./= nsim
	# zlnqx_mnt ./= nsim
	view(mntvec, mnt_range[12]) ./= nsim
	view(mntvec, mnt_range[13]) ./= nsim
	lftshare ./= nsim
	mktshare ./= nsim
	lnqrnd_lftj ./= nsim
	lnqrnd_migj ./= nsim

	copyto!(view(mntvec, mnt_range[1]), lftshare)
	uc_lft_pr = dot(lftshare, mktshare)

	# --- moments for XM: E(xm) ---
	mul!(view(mntvec, mnt_range[2]), xm, mktshare)
	# mul!(xm_mnt, xm, mktshare)

	# --- moments for housing price lnP: E(lnp) ---
	view(mntvec, mnt_range[3]) = dot(lnp, mktshare)
	# mlnp = dot(lnp, mktshare)

	# --- moments for XF (lambda): E(xf) & E(xf|k) ---
	# mul!(xf_mnt, xf, mktshare)
	mul!(view(mntvec, mnt_range[4]), xf, mktshare)
	broadcast!(*, nalt_tmp, lftshre, mktshare)
	# mul!(xf_mnt_lft, xf, nalt_tmp)
	mul!(view(mntvec, mnt_range[5]), xf, mktshare)
	# xf_mnt_lft ./= pr_lft
	view(mntvec, mnt_range[5]) ./= pr_lft

	# --- moments for income lnW: E(lnw) & E(lnw|k) ---
	# mlnw = dot(lnw, mktshare)
	# mlnw_lft = dot(lnw, nalt_tmp) / pr_lft
	view(mntvec, mnt_range[6]) = dot(lnw, mktshare)
	view(mntvec, mnt_range[7]) = dot(lnw, nalt_tmp) / pr_lft

	# --- moments for XL: E(xl|k) ---
	# xl_mnt .= (uc_lft_pr / pr_lft) * xl
	BLAS.axpy!((uc_lft_pr / pr_lft), xl, view(mntvec, mnt_range[8]))

	# --- moments for XT: E(xt|k) ---
	# xt_mnt .= (uc_lft_pr / pr_lft) * xt
	BLAS.axpy!((uc_lft_pr / pr_lft), xt, view(mntvec, mnt_range[9]))

	# --- E(xq'lnq | k = 0), E(lnw lnq | k = 0) ---
	broadcast!((x, y, z) -> x + y / (unit - z), lnq_mig, lnq_mig, lnqrnd_migj, pr_lft_alt)
	broadcast!(*, nalt_tmp, lnq_mig, mktshare)
	# mul!(xq_lnq_mig, xq, nalt_tmp)
	mul!(view(mntvec, mnt_range[14]), xq, nalt_tmp)
	# lnwq_mig = dot(lnw, nalt_tmp)
	view(mntvec, mnt_range[16]) = dot(lnw, nalt_tmp)

	# --- E(lnq^2 | k = 0) ---
	# lnq2_mig = dot(lnq_mig, nalt_tmp)
	view(mntvec, mnt_range[18]) = dot(lnq_mig, nalt_tmp)

	# --- E(xq'lnq | k = 1), E(lnw lnq | k = 1) ---
	broadcast!((x, y, z) -> x + y / z, lnq_lft, lnq_lft, lnqrnd_lftj, pr_lft_alt)
	broadcast!(*, nalt_tmp, lnq_lft, mktshare)
	# mul!(xq_lnq_lft, xq, nalt_tmp)
	mul!(view(mntvec, mnt_range[15]), xq, nalt_tmp)
	# lnwq_lft = dot(lnw, nalt_tmp)
	view(mntvec, mnt_range[17]) = dot(lnw, nalt_tmp)

	# --- E(lnq^2 | k = 1) ---
	# lnq2_lft = dot(lnq_lft, nalt_tmp)
	view(mntvec, mnt_range[19]) = dot(lnq_lft, nalt_tmp)

	# vcat(lftshare, xm_mnt, mlnp, xf_mnt, xf_mnt_lft, mlnw, mlnw_lft, xl_mnt, xt_mnt,
	# 	 zm_mnt_mig, zm_mnt_lft, zlnw_mnt, zlnqx_mnt, xq_lnq_mig, xq_lnq_lft, lnwq_mig,
	# 	 lnwq_lft, lnq2_mig, lnq2_lft)
end

function unpack_parm(parm, XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					 XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					 XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
					 xdim::Int, nalt::Int) where T <: AbstractFloat
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
	 bq = parm[(nxt + nxl + nxm + nxf + 4):(nxt + nxl + nxm + nxf + nxq + 3)]

	 bz = parm[(nxt + nxl + nxm + nxf + nxq + 4):(nxt + nxl + nxm + nxf + nxq + nzr + 3)] #<- observed household char.
	 sigu = exp(parm[nxt + nxl + nxm + nxf + nxq + nzr + 4])

	 mnt_idx = [nalt, nxm, 1, nxf, nxf, 1, 1, nxl, nxt, nzr, nzr, nzr, nzr, nxq, nxq, 1, 1, 1, 1]
	 return (bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu, mnt_idx)
end

function get_mnt_range(mnt_idx::AbstractVector{T}) where T <: Int
	mnt_end = cumsum(mnt_idx)
	mnt_bgn = [1; mnt_end[1:end-1] .+ 1]
	mnt_range = [mnt_bgn[i]:mnt_end[i] for i = 1:length(mnt_idx)]

	return mnt_range
end
