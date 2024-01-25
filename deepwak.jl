#clustering params
d = 14 # embedding size
l_e = [58,29,d] # encoder layer sizes
l_p = 5 # number of classifier layers
h = 5 # number of heads
c = 14 # maximum clusters per head
frac = 10 # test set fraction

include("julia/DeePWAK.jl")
include("julia/LayerFns.jl")

using CSV, DataFrames, StatsPlots
using JLD2

# training hyperparams
epochs = 1000
η = 0.001 #learning rate
λ = 0.001 #decay rate
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

groups = (DataFrame ∘ CSV.File)("data/groups.csv",normalizenames=true);

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

X = scaledat(dat');
m,n = size(X)
l_e = vcat(m,l_e)

n_test = Integer(2^round(log2(n) - log2(frac)))
n_train = n - n_test
test,train = sampledat(X,n_test) |> gpu
X = gpu(X);

loader = Flux.DataLoader((train,train),batchsize=n_test,shuffle=true) 

M = Chain(Parallel(vcat,map(_->DeePWAK(mlp(l_e,tanh),
                                  Chain(mlp4x(m,c,l_p),softmax),
                                  mlp(reverse(l_e),tanh)),
                       1:h)...),Dense(h*m => m)) |> gpu;

L = train!(M,loader,opt,test,epochs)
L = hcat(map(vcat,vcat(L...)...)...)
Tables.table(L) |> CSV.write("data/loss.csv")

M_s = Flux.state(M) |> cpu;
jldsave("DeePWAKBlock.jld2"; M_s)

C = mapreduce(vcat,M[1].layers) do m
    return (clusts ∘ m.partitioner)(X)
end

Tables.table(C') |> CSV.write("data/clusts.csv")

map(M[1].layers,1:h) do m,i
    E = m.encoder(X)
    Tables.table(E') |> CSV.write("data/embedding/$i.csv")
end

map(M[1].layers,1:h) do m,i
    C = m.partitioner(X)
    Tables.table(C') |> CSV.write("data/clust/$i.csv")
end

p_cat = Parallel(vcat,map(m->m.partitioner,M[1].layers)) |> gpu;
p_consensus = Chain(mlp4x(h*c,c,l_p),softmax) |> gpu

function loss_Cc(M,p_cat,p_consensus,X)
    E = p_cat(X)
    C = p_consensus(E)
    P = C' * C
    G = wak(P)
    M = Chain(Parallel(vcat,map(m->Chain(m.encoder,E->(G * E')',
                                m.decoder),
                       M[1].layers)...),M[2])
    return Flux.mse(M(X),X)
end

L_Cc = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        loss = m->loss_Cc(M,p_cat,m,x)
        l_test = loss_Cc(M,p_cat,p_consensus,test)
        l_train = update!(p_consensus,loss,opt)
        return l_train,l_test
    end
end

hcat(map(vcat,vcat(L_Cc...)...)...) |> Tables.table |> CSV.write("data/loss_Cc.csv")

C_c = (p_consensus ∘ p_cat)(X)
Tables.table(C_c') |> CSV.write("data/C_consensus.csv")
clusts(C_c)' |> Tables.table |> CSV.write("data/clusts_consensus.csv")

e_cat = Parallel(vcat,map(M->M.encoder,M[1].layers)) |> gpu
e_consensus = mlp([h*d,35,d],tanh) |> gpu

function loss_Ec(M,e_cat,e_consensus,X)
    M = Chain(Parallel(vcat,map(m->DeePWAK(Chain(e_cat,e_consensus),
                                  m.partitioner,
                                  m.decoder),
                       M[1].layers)...),M[2]) |> gpu
    return Flux.mse(M(X),X)
end

L_Ec = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        loss = m->loss_Ec(M,e_cat,m,x)
        l_test = loss_Ec(M,e_cat,e_consensus,test)
        l_train = update!(e_consensus,loss,opt)
        return l_train,l_test
    end
end
hcat(map(vcat,vcat(L_Ec...)...)...) |> Tables.table |> CSV.write("data/loss_Ec.csv")

e_s = Flux.state(e_consensus) |> cpu;
jldsave("e_consensus.jld2"; e_s)

E_c = (e_consensus ∘ e_cat)(X)
Tables.table(E_c') |> CSV.write("data/E_consensus.csv")
