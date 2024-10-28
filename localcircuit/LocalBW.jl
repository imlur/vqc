struct LocalBW <: LocalCircuit
	ts::LocalTensors
	flipped::Bool
end

process_args(::Type{LocalBW}, b::Bool) = (b,)

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
		return base + (b ? div(q - 1, 2) : div(q + 1, 2))
	end
end
matsize(::Type{LocalBW}, q, d, b::Bool) = get_row(q), q == 1 ? 1 : d
get_row(q) = q % 2 == 1 ? div(q + 1, 2) : div(q, 2)
function nth_tensor_idx(::Type{LocalBW}, l, i)
	q, flip = l.ts.q, l.flipped
	if q == 1
		return (1, 1)
	elseif q % 2 == 0
		return (i%div(q, 2)+1, div(i-1,div(q, 2))+1)
	else
		dbase, irem = div(i - 1, q) * 2 + 1, (i - 1) % q + 1
		thre = flip ? div(q - 1, 2) : div(q + 1, 2)
		dinc = irem > thre
		return (irem - dinc * thre, dbase + Int(dinc))
	end
end

function isvalidind(::Type{LocalBW}, l, i, j, b::Bool) 
	q, d = l.ts.q, l.ts.d
	row, col = size(l.ts.tensors)
	!(q % 2 == 1 && i == row && ((b && j % 2 == 1) || (!b && j % 2 == 0)))
end

# Example : LocalBW(4, 7) - layer1 : 3 gates, layer2 : 2 gates 
process_args(::Type{LocalBW}, b::Bool) = (b, )

function gettag(::Type{LocalBW}, l, i, j, flip::Bool)
	row, col = size(l.ts.tensors); q = l.ts.q
	layer2 = (flip && j % 2 == 1) || (!flip && j % 2 == 0)
	if layer2 # div(q , 2) gates
		t1, t2 = coord(i, j, i, j-1), coord(i, j, i+1, j-1)
		t3, t4 = coord(i, j, i+1, j+1), coord(i, j, i, j+1)
		if i == row && q % 2 == 0
			t2, t3 = coord(i, j, i, j-2), coord(i, j, i, j+2)
		end
	else # q is even -> q / 2 gates, q is odd -> (q + 1) / 2 gates
		t1, t2 = coord(i, j, i-1, j-1), coord(i, j, i, j-1)
		t3, t4 = coord(i, j, i, j+1), coord(i, j, i-1, j+1)
		if i == 1
			t1, t4 = coord(i, j, i, j-2), coord(i, j, i, j+2)
		end
		if i == row && q % 2 == 1
			t2, t3 = coord(i, j, i, j-2), coord(i, j, i, j+2)
		end
	end
	
	# For cases where jth column is at edge of matrix
	if j == 1
		if layer2
			t1, t2 = "in,q$(2*i)", "in,q$(2*i+1)"
		else
			t1, t2 = "in,q$(2*i-1)", "in,q$(2*i)"
		end
	elseif j == 2 && layer2 && i == row && q % 2 == 0
		t2 = "in,q$(q+1)"
	elseif j == 2 && !layer2 && i == row && q % 2 == 1
		t2 = "in,q$(q+1)"
	elseif j == 2 && !layer2 && i == 1
		t1 = "in,q1"
	end

	if j == col
		if layer2
			t3, t4 = "out,q$(2*i+1)", "out,q$(2*i)"
		else
			t3, t4 = "out,q$(2*i)", "out,q$(2*i-1)"
		end
	elseif j == col-1 && layer2 && i == row && q % 2 == 0
		t3 = "out,q$(q+1)"
	elseif j == col-1 && !layer2 && i == row && q % 2 == 1
		t3 = "out,q$(q+1)"
	elseif j == col-1 && !layer2 && i == 1
		t4 = "out,q1"
	end
	return (t1, t2, t3, t4)
end
