using StatsFuns:logistic, log1pexp
function mig_leftbh_llk(parm, Delta, yl, ym, lnW, lnP, XT, XL, XM, XF, XQ,
						ZSHK, USHK, nind, nalt, nsim, ngvec::AbstractVector{Int})
	##
	## Delta:   nalt x g Matrix
	## XT: 		nt x N Matrix
	## XL: 		nl x N Matrix
	## XM: 		nm x (NJ) Matrix
	## XF: 		nf x (NJ) Matrix
	## XQ: 		nq x N Matrix
	## ZSHK: 	nz x (NS) Matrix
	## USHK:	NS Vector
	## ngvec:	g Vector
	##

	bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu = unpack_parm(parm, XT, XL, XM, XF, XQ, ZSHK)

	# --- key input vectors ---
	theta = XT' * bt
	broadcast!(logistic, theta, theta)
	Xbl = XL' * bl
	Xbq = XQ' * bq

	Xbm = XM' * bm
	ln1mlam = XF' * bf
	broadcast!(log1pexp, ln1mlam, ln1mlam) #<-- [-ln(1-lam)]

	Zbr = ZSHK' * bz

	# --- begin the loop ---
	llk = zero(eltype(parm))
	for i = 1:nind
		ind_sel = (1 + nalt * (i - 1)):(i * nalt)
		sim_sel = (1 + nsim * (i - 1)):(i * nsim)
		g = locate_gidx(i, ngvec)

		llk += individual_llk(bw, blft, bitr, sigu, view(Delta, :, g), view(yl, ind_sel),
							  view(ym, ind_sel), view(lnW, ind_sel), view(lnP, ind_sel),
							  xbt[i], xbl[i], view(xbm, ind_sel), view(ln1mlam, ind_sel),
							  xbq[i], view(zbr, sim_sel), view(ushk, sim_sel), nalt, nsim)
	end

	return llk
end


using StatsFuns:logistic
function individual_llk(bw, blft, bitr, sigu, delta, yl, ym, lnw, lnp, xbt, xbl,
						xbm, ln1mlam, xbq, zbr, ushk, nalt, nsim)
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

	# --- calculate (dlnq, lnq_mig) ---
	TT = eltype(bw)
	dlnq = zeros(TT, nalt)
	lnq_mig = zeros(TT, nalt)
	lnq_alt!(lnq_mig, dlnq, lnw, ln1mlam, xbq, bw, blft, bitr)

	# --- setup containers ---
	unit = one(TT)
	loc_sel = (yl .== unit)
	lft_pr_js = zeros(TT, nalt)
	loc_pr_js = zeros(TT, nalt)
	# llki = zeros(TT, nalt)
	llki = zero(TT)

	# --- begin the loop ---
	for s = 1:nsim
		zrnd = zbr[s]
		urnd = ushk[s]
		for j = 1:nalt
			# calculate lft_prob
			theta = logistic(xbt + zrnd + sigu * urnd)
			lft_pr_tmp = leftbh_prob(theta, ln1mlam[j], xbl, dlnq[j])
			lft_pr_js[j] = lft_pr_tmp * ym[j] + (unit - ym[j]) * (unit - lft_pr_tmp)

			# calculate location prob
			gambar = gamfun(lnw[j], dlnq[j], lnq_mig[j], xbl, ln1mlam[j], theta)

			# --- location specific utility ---
			loc_pr_js[j] = Vloc(xbm[j], gambar, delta[j])

		end
		emaxprob!(loc_pr_js)
		# llki .+= (loc_pr_js .* yl) .* lft_pr_js
		llki += view(loc_pr_js, loc_sel) * view(lft_pr_js, loc_sel)
	end #<-- end of s loop

	return -log(llki / nsim)
end


using StatsFuns:logistic
function unpack_parm(parm, XT, XL, XM, XF, XQ, ZSHK)
	 nxt = size(XT, 1)
	 nxl = size(XL, 1)
	 nxm = size(XM, 1)
	 nxf = size(XF, 1)
	 nxq = size(XQ, 1)
	 nzr = size(ZSHK, 1)

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

	 return (bw, blft, bitr, bt, bl, bm, bf, bq, bz, sigu)
end
