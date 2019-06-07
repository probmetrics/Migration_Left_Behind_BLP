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

function mig_leftbh_llk(parm, lnW, lnP, XT, XL, XM, XF, XQ, ZSHK, USHK)
	##
	## XT: 		nt x N Matrix
	## XL: 		nl x N Matrix
	## XM: 		nm x (NJ) Matrix
	## XF: 		nf x (NJ) Matrix
	## XQ: 		nq x N Matrix
	## ZSHK: 	nz x (NS) Matrix
	## USHK:	NS Vector
	##


end
