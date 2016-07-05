local torch = require 'torch'
local argcheck = require 'argcheck'

local M = {}

local pgdCheck = argcheck{
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
   local args = pgdCheck(...)
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

return M
