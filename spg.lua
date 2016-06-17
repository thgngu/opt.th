local torch = require 'torch'
local argcheck = require 'argcheck'

local M = {}

local spgCheck = argcheck{
   pack=true,
   -- TODO: Comments
   {name='x0', type='torch.*Tensor'},
   {name='f', type='function'},
   {name='g', type='function'},
   {name='proj', type='function'},
   {name='m', type='number', default=10, opt=true},
   {name='eps', type='number', default=1e-6, opt=true},
   {name='maxit', type='number', default=1000, opt=true}
}
function M.solve(...)
   local args = spgCheck(...)
   local results = {}

   local x0 = args.x0
   local f = args.f
   local g = args.g
   local proj = args.proj
   local m = args.m
   local eps = args.eps
   local maxit = args.maxit

   local alpha_min = 1e-3
   local alpha_max = 1e3

   local f_hist = torch.Tensor(maxit):typeAs(x0)

   local linesearchCheck = argcheck{
      pack=true,
      -- TODO: Comments
      {name='x_k', type='torch.*Tensor'},
      {name='g_k', type='torch.*Tensor'},
      {name='d_k', type='torch.*Tensor'},
      {name='k', type='number'}
   }
   local function linesearch(...)
      local args = linesearchCheck(...)
      local x_k = args.x_k
      local g_k = args.g_k
      local d_k = args.d_k
      local k = args.k

      local gamma = 1e-4
      local sigma_1 = 0.1
      local sigma_2 = 0.9

      local f_max = torch.max(f_hist[{{math.max(1, k-m+1), k}}])
      local delta = torch.dot(g_k, d_k)

      local x_p = x_k + d_k
      local lambda = 1

      local f_p = f(x_p)
      while f_p > f_max + gamma*lambda*delta do
         local lambda_t = 0.5*(lambda^2)*delta/(f_p-f_k-lambda*delta)
         if lambda_t >= sigma_1 and lambda_t <= sigma_2*lambda then
            lambda = lambda_t
         else
            lambda = lambda/2.0
         end
         x_p = x_k + torch.mul(d_k, lambda)
         f_p = f(x_p)
      end
      return lambda
   end

   -- If x_0 \not\in \Omega, replace x_0 by P(x_0)
   local x = proj(x0)
   local g_new = g(x)
   local d = proj(x - g_new) - x
   local alpha = math.min(alpha_max, math.max(alpha_min, 1/torch.max(d)))

   results.bestF = nil
   results.bestX = x:clone()

   for k = 1, maxit do
      local f_k = f(x)
      local g_k = g_new
      f_hist[k] = f_k
      if results.bestF == nil or f_k < results.bestF then
         results.bestF = f_k
         results.bestX:copy(x)
      end
      local d = proj(x - torch.mul(g_k, alpha)) - x
      if d:norm(2) < eps then
         break
      end
      local lambda = linesearch(x, g_k, d, k)
      local s = torch.mul(d, lambda)
      x:add(s)
      g_new = g(x)
      local y = g_new - g_k
      local beta = torch.dot(s, y)
      if beta < 0 then
         alpha = alpha_max
      else
         alpha = math.min(alpha_max, math.max(alpha_min, torch.dot(s, s)/beta))
      end
   end

   return results
end

return M
