using DataFrames
function data_prepare(df::AbstractDataFrame; trs::Bool = false)
    ndt = nrow(df)
    city_alts = Vector{Int64}(df[:, :city_alts])
    nalt = length(unique(city_alts))
    ngvec = by(view(df, df[:, :chosen] .== 1, :), :year, nrow, sort = true)[:, 2]
    ngvec = cumsum(ngvec)
    nind = Int(ndt / nalt)
    dgvec = [locate_gidx(i, ngvec) for i = 1:nind]

    lnW = Vector{Float64}(df[:, :lnhinc_alts])
    lnW = lnW .- mean(lnW, weights(Vector{Float64}(df[:, :w_l])))
    lnP = Vector{Float64}(df[:, :lnhprice])

    wgt = Vector{Float64}(df[df[:, :chosen] .== 1, :w_l])

    sgwgt = countmap(dgvec, weights(wgt))
    sgwgt = [sgwgt[i] for i = 1:length(sgwgt)]

    # --- initial value of delta ---
    cfreq = by(view(df, df[:, :chosen] .== 1, :), [:year, :city_alts], d -> sum(d[:, :w_l]))
    cfreq = reshape(sort(cfreq, [:year, :city_alts])[:, :x1], nalt, length(ngvec))
    cprop = cfreq ./ sum(cfreq, dims = 1)
    lnDataShare = log.(cprop)
    Delta_init = lnDataShare .- lnDataShare[1, :]'

    # --- household preference ---
    XTnames = [:highsch_f, :highsch_m, :age_f, :age_m, :han]
    XT = [ones(nind) convert(Array{Float64, 2}, df[df[:, :chosen] .== 1, XTnames])]

    # --- migration cost ---
    XMnames = [:lndist, :cross_prov, :cross_regn, :lndist_crsprov, :lndist_crsregn,
               :amenity_pca_flowdur, :migdist_flowdur, :amenity_pca_highsch_f,
               :migdist_highsch_f, :amenity_pca_highsch_m, :migdist_highsch_m,
               :amenity_pca_age_f, :migdist_age_f, :amenity_pca_age_m, :migdist_age_m]
    XM = convert(Array{Float64, 2}, df[:, XMnames])

    # --- left-behind utility loss ---
    XLnames = [:cfemale, :nchild, :cagey, :cageysq]
    XL = convert(Array{Float64, 2}, df[df[:, :chosen] .== 1, XLnames])

    # --- fixed cost ---
    # NOTE: the most critical part of the model!!
    XFnames = [:htreat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat,
               :lnmnw_city, :nchild_lnhp]
    XF = [ones(ndt) convert(Array{Float64, 2}, df[:, XFnames])]

    # --- cognitive ability ---
    # TODO: how to incorporate regional eduation quality?
    #       we have identification problem.

    XQnames = [:cfemale, :cagey, :nchild]
    XQ = convert(Array{Float64, 2}, df[df[:, :chosen] .== 1, XQnames])

    XQJnames = [:tstu2_ratio, :sschool_per]
    XQJ_mig = convert(Array{Float64, 2}, df[:, XQJnames])

    if trs == true
        XT = copy(XT')
        XL = copy(XL')
        XM = copy(XM')
        XF = copy(XF')
        XQ = copy(XQ')
        XQJ_mig = copy(XQJ_mig')
    end

    return(lnDataShare, Delta_init, lnW, lnP, XQJ_mig,
           wgt, sgwgt, XT, XM, XL, XF, XQ, nalt, nind, dgvec)
end

#=
XQCnames = [:tstu1_ratio, :tstu2_ratio]
XQC = convert(Array{Float64, 2}, LeftbhData[XQCnames])
XQC = XQC .- mean(XQC, weights(Vector{Float64}(LeftbhData[:cw_age9])), 1)
XQC = SharedArray(copy(XQC'))
=#
