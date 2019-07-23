using StatsFuns:logistic, log1pexp
function mig_leftbh_llk(parm, Delta::AbstractMatrix{T}, YL::AbstractVector{T},
						YM::AbstractVector{T}, lnW::AbstractVector{T},
						lnP::AbstractVector{T}, QXJ_mig::AbstractVector{T},
						QXJ_lft::AbstractVector{T},	XT::AbstractMatrix{T},
						XL::AbstractMatrix{T}, XM::AbstractMatrix{T},
						XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
						ZSHK::AbstractMatrix{T}, USHK::AbstractVector{T},
						wgt::AbstractVector{T}, nind::Int, nalt::Int,
						nsim::Int, dgvec::AbstractVector{Int};
						alpha::AbstractFloat = 0.12, xdim::Int = 2) where T <: AbstractFloat
	##
	## Delta:   nalt x g Matrix
	## XT: 		nt x N Matrix
	## XL: 		nl x N Matrix
	## XM: 		nm x (NJ) Matrix
	## XF: 		nf x (NJ) Matrix
	## XQ: 		nq x NJ Matrix
	## ZSHK: 	nz x (NS) Matrix
	## USHK:	NS Vector
	## dgvec:	N Vector
	##

	bw, blft, bitr, bqxj, bt, bl, bm, bf, bq, bz, sigu =
			unpack_parm(parm, XT, XL, XM, XF, XQ, ZSHK; xdim = xdim)

	# --- key input vectors ---
	Xbt = XT * bt
	Xbl = XL * bl

	Xbm = XM * bm
	ln1mlam = XF * bf
	broadcast!(log1pexp, ln1mlam, ln1mlam) #<-- [-ln(1-lam)]
	dlnQ = blft .+ bitr * lnW + bw * ln1mlam + bqxj * (QXJ_lft - QXJ_mig)
	lnQ_mig = bw * lnW - bw * ln1mlam + XQ * bq + bqxj * QXJ_mig

	Zbr = ZSHK * bz

	# --- begin the loop ---
	llk = zero(eltype(parm))
	@fastmath @inbounds @simd for i = 1:nind
		ind_sel = (1 + nalt * (i - 1)):(i * nalt)
		sim_sel = (1 + nsim * (i - 1)):(i * nsim)
		g = view(dgvec, i)

		llk += individual_llk(bw, blft, bitr, sigu, alpha, view(Delta, :, g), view(YL, ind_sel),
							  view(YM, ind_sel), view(lnW, ind_sel), view(lnP, ind_sel),
							  Xbt[i], Xbl[i], view(Xbm, ind_sel), view(ln1mlam, ind_sel),
							  view(dlnQ, ind_sel), view(lnQ_mig, ind_sel), view(Zbr, sim_sel),
							  view(USHK, sim_sel), nalt, nsim) * wgt[i]
	end

	return llk
end


using StatsFuns:logistic
function individual_llk(bw, blft, bitr, bqxj, sigu, alpha, delta, yl, ym, lnw,
						lnp, xbt, xbl, xbm, ln1mlam, dlnq, lnq_mig, zbr, ushk,
						nalt, nsim)
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

	# --- setup containers ---
	TT = promote_type(eltype(bw), eltype(lnw))
	unit = one(TT)
	llki = zero(TT)

	# --- begin the loop ---
	@fastmath @inbounds @simd for s = 1:nsim
		lftpr_s = zero(TT)
		llks = zero(TT)
		eVsums = zero(TT)

		zrnd = zbr[s]
		urnd = ushk[s]
		theta = logistic(xbt + zrnd + sigu * urnd)
		@inbounds for j = 1:nalt
			# calculate lft_prob
			lft_pr_tmp = leftbh_prob(theta, ln1mlam[j], xbl, dlnq[j])
			lftpr_s += (lft_pr_tmp * ym[j] + (unit - ym[j]) * (unit - lft_pr_tmp)) * yl[j]

			# calculate location prob
			gambar = gamfun(lnw[j], dlnq[j], lnq_mig[j], xbl, ln1mlam[j], theta)

			# --- location specific utility ---
			Vj = Vloc(alpha, lnp[j], theta, xbm[j], gambar, delta[j])
			llks += Vj * yl[j]
			eVsums += exp(Vj)
		end
		llki += llks - log(eVsums) + log(lftpr_s)

	end #<-- end of s loop

	return -llki / nsim
end


using StatsFuns:logistic
function unpack_parm(parm, XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					 XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					 XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T};
					 xdim::Int = 2) where T <: AbstractFloat
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

	 return (bw, blft, bitr, bqxj, bt, bl, bm, bf, bq, bz, sigu)
end
