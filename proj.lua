local torch = require 'torch'

local M = {}

function M.boxInPlace(x, l, u)
   local I = torch.lt(x, l)
   x[I] = l
   I = torch.gt(x, u)
   x[I] = u
   return x
end

return M
