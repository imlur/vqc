abstract type QMPS{T} end

struct QMPSBlocks{T}
	locals::Vector{T}
	params::Vector{Array{NTuple{6, Float64}, 2}}
	ssize::Int
	qb::Int
	dep::Int
	idx::Index
end

# g[i] when g::QMPS : ith block of g
Base.getindex(g::QMPS, i::Int) = g.blk.locals[i]
for f in fieldnames(QMPSBlocks)	@eval $f(g::QMPS) = g.blk.$f end

check_qd(q, d, g::QMPS) = @assert q == qb(g) && d > dep(g)

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
	T(QMPSBlocks{localtype(T)}(blocks, params, L, q, d, ii))
end

nblocks(g::QMPS) = nblocks(typeof(g), ssize(g), qb(g), dep(g))
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
getithblock(g::T, i) where {T<:QMPS} = g[i]

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

function getinit(g::QMPS) 
	ind = idx(g)
	itgs = itags(g[1])
	inds = [settags(ind, tag) for tag in itgs]
	tensor = ITensor(0.0, inds...)
	tensor[[1 for _=1:length(inds)]...] = 1.0
	return tensor
end
contract(g::QMPS) = contract(g, getinit(g))
prepnextcont(g::QMPS, t::ITensor, i) = prepnextcont(typeof(g), g, t, i)
function contract(g::QMPS, t::ITensor)
	for (i, blk) in enumerate(g.blk.locals)
		t = contract(blk, t)
		if i < nblocks(g)
			t = prepnextcont(g, t, i)
		end
	end
	return t
end

function Base.show(io::IO, ::MIME"text/plain", g::T) where {T<:QMPS}
	print(T, ", L = $(ssize(g)), q = $(qb(g)), d = $(dep(g))")
	println()
	nb = nblocks(g)
	for i=1:nb
		println()
		print("Block $(i): \t")
		Base.show(io, "text/plain", g[i])
	end
end


include("QMPS_1Block.jl")
include("QMPS_LBlocks.jl")
