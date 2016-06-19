package = "spg"
version = "scm-1"

source = {
   url = "git://github.com/bamos/spg.torch.git",
   tag = "master"
}

description = {
   summary = "Spectral Projected Gradient (SPG) Torch implementation for convex-constrained optimization.",
   detailed = "",
   homepage = "https://github.com/bamos/spg.torch"
}

dependencies = {
   "argcheck",
   "torch >= 7.0"
}

build = {
   type = "builtin",
   modules = {
      ["spg"] = "spg.lua",
      ["spg.proj"] = "proj.lua",
   }
}