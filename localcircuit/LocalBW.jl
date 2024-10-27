struct LocalBW <: LocalCircuit
	ts::LocalTensors
	flipped::Bool
end

process_args(::Type{LocalBW}, b::Bool) = (b,)

# TODO: complete this struct (initialization, tags, ...)
LocalBW(q) = LocalBW(LocalTensors(Matrix{ITensor}(undef, 0, 0),
	Index(0), q, 0, 0), true)
LocalBW(args...) = createlocal(LocalBW, args...)
function ntensors(::Type{LocalBW}, q, d, b::Bool)
	if q == 1
		return (1, 1)
	elseif q % 2 == 0 || d % 2 == 0
		return div(q * d, 2)
	else
		base = div(q * (d - 1), 2)
		return base + b ? div(q - 1, 2) : div(q + 1, 2)
	end
end
matsize(::Type{LocalBW}, q, d, b::Bool) = get_row(q), d
get_row(q) = q % 2 == 1 ? div(q + 1, 2) : div(q, 2)
function nth_tensor_idx(::Type{LocalBW}, l, i, b::Bool)
	q = l.ts.q
	if q == 1
		return (1, 1)
	elseif q % 2 == 0
		return (i%div(q, 2)+1, div(i-1,div(q, 2))+1)
	else
		dbase, irem = div(i - 1, q) * 2 + 1, (i - 1) % q + 1
		thre = b ? div(q - 1, 2) : div(q + 1, 2)
		dinc = irem > thre
		return (irem - dinc * thre, d + Int(dinc))
	end
end

function isvalidind(::Type{LocalBW}, l, i, j, b::Bool) 
	q, d = l.ts.q, l.ts.d
	row, col = size(l.ts.tensors)
	!(q % 2 == 1 && i == row && ((b && j % 2 == 1) || (!b && j % 2 == 0)))
end
	
process_args(::Type{LocalBW}, b::Bool) = (b, )
function gettag(::Type{LocalBW}, obj, i, j, l::LocalBW, b::Bool)
	if q == 1
		return ("in,q1", "in,q2", "out,q2", "out,q1")
	else
	end
end
