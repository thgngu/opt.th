package = "cvx-optim"
version = "scm-1"

source = {
   url = "git://github.com/bamos/cvx-optim.torch.git",
   tag = "master"
}

description = {
   summary = "Convex optimization library",
   detailed = "",
   homepage = "https://github.com/bamos/cvx-optim.torch"
}

dependencies = {
   "argcheck",
   "torch >= 7.0"
}

build = {
   type = "builtin",
   modules = {
      ["cvx-optim"] = "init.lua",
      ["cvx-optim.spg"] = "spg.lua",
      ["cvx-optim.proj"] = "proj.lua",
   }
}