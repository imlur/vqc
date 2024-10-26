struct LocalLadder <: LocalCircuit
	ts::LocalTensors
end

function tlin(q, d, i, j)
	if j == 1
		if i == 1
			t1, t2 = "in,q1", "in,q2"
		else
			t1, t2 = "$(i)-$(j),$(i-1)-$(j)", "in,q$(i+1)"
		end
	else
		if i == 1
			t1, t2 = "$(i)-$(j),$(i)-$(j-1)", "$(i)-$(j),$(i+1)-$(j-1)"
		else
			t1, t2 = "$(i)-$(j),$(i-1)-$(j)", "$(i)-$(j),$(min(i+1, q))-$(j-1)"
		end
	end
	return (t1, t2)
end

function tlout(q, d, i, j)
	if j == d
		if i == q
			t1, t2 = "out,q$(q+1)", "out,q$(q)"
		else
			t1, t2 = "$(i)-$(j),$(i+1)-$(j)", "out,q$(i)"
		end
	else
		if i == q
			t1, t2 = "$(i)-$(j),$(i)-$(j+1)", "$(i)-$(j),$(i-1)-$(j+1)"
		else
			t1, t2 = "$(i)-$(j),$(i+1)-$(j)", "$(i)-$(j),$(max(i-1, 1))-$(j+1)"

		end
	end
	return (t1, t2)
end


LocalLadder(q) = LocalLadder(LocalTensors(Matrix{ITensor}(undef, 0, 0), 
	Index(0), q, 0, 0))
LocalLadder(args...) = createlocal(LocalLadder, args...)

ntensors(::Type{LocalLadder}, q, d) = q * d
matsize(::Type{LocalLadder}, q, d) = q, d
nth_tensor_idx(::Type{LocalLadder}, l, i) = (i%l.ts.q+1, div(i-1,l.ts.q)+1)

process_args(::Type{LocalLadder}) = ()
gettag(::Type{LocalLadder}, q, d, i, j, l::LocalLadder) = 
	(tlin(q, d, i, j)..., tlout(q, d, i, j)...)
