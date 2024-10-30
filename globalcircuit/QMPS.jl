abstract type QMPS{T} end

# TODO: add initialization code similar to localcircuit structs

struct QMPSBlocks{T}
	locals::Vector{T}
	params::Vector{Array{NTuple{6, Float64}, 2}}
	L::Int
	q::Int
	d::Int
end

gettensor(g::QMPS, bidx, i, j) = g.blk.locals[bidx].ts.tensors[i, j]
check_qd(q, d, g::QMPS) = @assert q == g.blk.q && d > g.blk.d

createglobal(::Type{T}, L, q, d) where T =
	createglobal(T, L, q, d, Index(2))
createglobal(::Type{T}, L, q, d, ii::Index) where T =
	createglobal(T, L, q, d, ii, nothing)
createglobal(::Type{T}, L, q, d, g::T) where T =
	createglobal(T, L, q, d, Index(2), g)

localtype(::Type{<:QMPS{T}}) where {T <: LocalCircuit} = T

function createglobal(::Type{T}, L, q, d, ii::Index, g::Union{T, Nothing}) where {T <: QMPS}
	blocks = getblocks(T, L, q, d, ii, g)
	params = getparams(blocks)
	T(QMPSBlocks{localtype(T)}(blocks, params, L, q, d))
end

function getblocks(::Type{T}, L, q, d, ii, g::Union{T, Nothing}) where {T}
	nb = nblocks(T, L, q, d)
	S = localtype(T)
	blocks = Vector{S}(undef, nb)
	for i in 1:nb
		bq, bd = blocksize(T, L, q, d, i)
		aux = auxargs(T, L, q, d, i)
		itags = getitags(T, L, q, d, i)
		otags = getotags(T, L, q, d, i)
		blk = getithblock(g, i)
		blocks[i] = S(bq, bd, ii, blk, aux...; it=itags, ot=otags) 
	end
	return blocks
end

auxargs(::Any, L, q, d, i) = ()
getitags(::Any, L, q, d, i) = nothing
getotags(::Any, L, q, d, i) = nothing
getithblock(::Nothing, i) = nothing
getithblock(g::T, i) where {T<:QMPS} = g.blk.locals[i]

getparams(blocks::Vector{T}) where {T <: LocalCircuit} =
	[getparam(b) for b in blocks]

function getparam(b::T) where {T <: LocalCircuit}
	tensors = b.ts.tensors; row, col = size(tensors)
	param = Array{NTuple{6, Float64}}(undef, row, col)
	for i in 1:row
		for j in 1:col
			if isvalidind(T, b, i, j)
				param[i, j] = tensortoparam(tensors[i, j])
			end
		end
	end
	return param
end

function tensortoparam(i::ITensor)
	mat = reshape(i.tensor.storage[:], 4, 2, 2)
	l = log(reshape(permutedims(mat, [1, 3, 2]), 4, 4))
	return (l[5], l[9], l[13], l[10], l[14], l[15])
end

function paramtoarr(p::NTuple{6, Float64})
	A = zeros(4, 4)
	pidx = 1
	for i = 1:4
		for j = i+1:4
			A[i, j] = p[pidx]
			A[j, i] = -p[pidx]
			pidx += 1
		end
	end
	expo = reshape(exp(A), 2, 2, 2, 2)
	return permutedims(expo, [1, 2, 4, 3])
end

# TODO: implement contraction
# contract(

include("QMPS_1Block.jl")
include("QMPS_LBlocks.jl")
