local torch = require 'torch'
local argcheck = require 'argcheck'

local M = {}

local solveCheck = argcheck{
   pack=true,
   -- TODO: Comments
   {name='x0', type='torch.*Tensor'},
   {name='f', type='function'},
   {name='g', type='function'},
   {name='proj', type='function'},
   {name='acc', type='boolean', default=true, opt=true},
   {name='lambda', type='number', opt=true,
    help='Fixed step size. If not provided this is found with a line search.'},
   {name='eps', type='number', default=1e-6, opt=true},
   {name='maxit', type='number', default=1000, opt=true},
   {name='callback', type='function', opt=true},
}
function M.solve(...)
   local args = solveCheck(...)
   local results = {}

   local x0 = args.x0
   local f = args.f
   local g = args.g
   local proj = args.proj
   local acc = args.acc
   local lambda = args.lambda == nil and 1.0 or args.lambda
   local eps = args.eps
   local maxit = args.maxit
   local callback = args.callback

   results.feval = 0
   results.geval = 0

   local x = proj(x0)
   local prev_x = x

   results.bestF = nil
   results.bestX = x:clone()

   for k = 1, maxit do
      local omega = (k-1.0)/(k+2.0)
      local d = torch.csub(x, prev_x)
      local y = acc and torch.add(x, omega, d) or x
      if k > 1 and d:norm(2) < eps then
         break
      end
      local f_y = f(y)
      local g_y = g(y)
      prev_x = x
      local f_x
      if args.lambda == nil then
         -- Line search.
         local beta = 0.5
         while true do
            x = proj(torch.add(y, -lambda, g_y))
            local dxy = torch.csub(x, y)
            f_x = f(x)
            results.feval = results.feval + 1
            if f_x <= f_y + g_y:dot(dxy) + dxy:norm(2)^2/(2*lambda) then
               break
            end
            lambda = beta * lambda
         end
      else
         x = proj(torch.add(y, -lambda, g_y))
         f_x = f(x)
         results.feval = results.feval + 1
      end
      if results.bestF == nil or f_x < results.bestF then
         results.bestF = f_x
         results.bestX:copy(x)
      end
      if callback then callback(k, results.bestF, y, g_y, lambda) end
   end

   return results
end

local solveBatchCheck = argcheck{
   pack=true,
   -- TODO: Comments
   {name='x0s', type='torch.*Tensor'},
   {name='f', type='function'},
   {name='g', type='function'},
   {name='proj', type='function'},
   {name='acc', type='boolean', default=true, opt=true},
   {name='lambda', type='number', opt=true,
    help='Fixed step size. If not provided this is found with a line search.'},
   {name='eps', type='number', default=1e-6, opt=true},
   {name='maxit', type='number', default=1000, opt=true},
   {name='callback', type='function', opt=true},
}
function M.solveBatch(...)
   local args = solveBatchCheck(...)
   local results = {}

   local x0s = args.x0s
   local f = args.f
   local g = args.g
   local proj = args.proj
   local acc = args.acc
   local lambda = args.lambda == nil and 1.0 or args.lambda
   local eps = args.eps
   local maxit = args.maxit
   local callback = args.callback

   results.feval = 0
   results.geval = 0

   local xs = proj(x0s)
   local prev_xs = xs

   local nSamples = xs:size(1)

   results.bestFs = nil
   results.bestXs = xs:clone()

   for k = 1, maxit do
      local omega = (k-1.0)/(k+2.0)
      local ds = torch.csub(xs, prev_xs)
      local ys = acc and torch.add(xs, omega, ds) or xs
      if k > 1 and ds:norm(2) < eps then
         break
      end
      local f_ys = f(ys)
      local g_ys = g(ys)
      prev_xs = xs
      local f_xs
      if args.lambda == nil then
         -- Line search.
         assert(false, 'unimplemented')
         -- local beta = 0.5
         -- while true do
         --    x = proj(torch.add(y, -lambda, g_y))
         --    local dxy = torch.csub(x, y)
         --    f_x = f(x)
         --    results.feval = results.feval + 1
         --    if f_x <= f_y + g_y:dot(dxy) + dxy:norm(2)^2/(2*lambda) then
         --       break
         --    end
         --    lambda = beta * lambda
         -- end
      else
         xs = proj(torch.add(ys, -lambda, g_ys))
         f_xs = f(xs)
         results.feval = results.feval + 1
      end
      if results.bestF == nil then
         results.bestFs = f_xs
         results.bestXs:copy(xs)
      else
         local betterIdxs = f_xs:lt(results.bestFs)
         results.bestFs[betterIdxs] = f_xs[betterIdxs]
         local xs_flat = xs:view(nSamples, -1)
         local betterIdxsX = betterIdxs:view(-1,1):repeatTensor(1, xs_flat:size(2)):viewAs(xs)
         results.bestXs[betterIdxsX] = xs[betterIdxsX]
      end
      if callback then callback(k, results.bestF, y, g_y, lambda) end
   end

   return results
end


return M
