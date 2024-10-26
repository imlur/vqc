
abstract type QMPS{T} end

# TODO: add initialization code similar to localcircuit structs

struct QMPSBlocks{T}
	locals::Vector{T}
	params::Vector{Array{Float64, 3}}
	L::Int
	q::Int
	d::Int
end


include("QMPS_1Block.jl")
