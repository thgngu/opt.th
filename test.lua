#!/usr/bin/env th

local torch = require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

local coptim = require 'cvx-optim'

local tester = torch.Tester()
local cOptimTest = torch.TestSuite()

function cOptimTest.simple()
   -- driver1.f from SPG

   -- Dimension of the problem.
   local n = 10

   -- Bounds.
   local l = torch.Tensor(n):fill(-100.0)
   local u = torch.Tensor(n):fill(50.0)

   -- Initial point.
   local x0 = torch.Tensor(n):fill(60.0)

   local maxit = 1000
   local eps2 = 1e-6
   local m = 10

   local function f(x) return x:norm()^2 end
   local function g(x) return x:clone():mul(2) end

   local function proj(x)
      local r = x:clone()
      local I = torch.lt(r, l)
      r[I] = l[I]
      I = torch.gt(r, u)
      r[I] = u[I]
      return r
   end

   local results = coptim.spg.solve(x0, f, g, proj, m, eps2, maxit)

   tester:asserteq(results.bestF, 0, 'Invalid optimal value.')
   tester:assertTensorEq(results.bestX, torch.zeros(n), 1e-5, 'Invalid optimal location.')

   results = coptim.pgd.solve(x0, f, g, proj)

   tester:asserteq(results.bestF, 0, 'Invalid optimal value.')
   tester:assertTensorEq(results.bestX, torch.zeros(n), 1e-5, 'Invalid optimal location.')
end

tester:add(cOptimTest)
tester:run()
