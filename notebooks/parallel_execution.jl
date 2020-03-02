@everywhere begin
    function slow_add( x )
        sleep(1)
        return x + 1
    end
end
@time slow_add(2)

Threads.nthreads()

using Distributed
using SharedArrays

rdd = SharedArray{Int64}(4)
@distributed for i in eachindex(rdd)
    rdd[i] = slow_add(i)
end

@time rdd

@time results = [ slow_add(i) for i in 1:4]

@time futures = [ @spawnat :any slow_add(i) for i in 1:4]

@time results = [ fetch(futures[i]) for i in 1:4]

# +
a = zeros(10)

Threads.@threads for i = 1:10
           a[i] = Threads.threadid()
       end

# -


