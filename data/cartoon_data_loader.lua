--------------------------------------------------------------------------------
-- Subclass of BaseDataLoader that provides data from two datasets.
-- The samples from the datasets are not aligned.
-- The datasets can have different sizes
--------------------------------------------------------------------------------
require 'data.base_data_loader'

local class = require 'class'
data_util = paths.dofile('data_util.lua')

CartoonDataLoader = class('CartoonDataLoader', 'BaseDataLoader')

function CartoonDataLoader:__init(conf)
  BaseDataLoader.__init(self, conf)
  conf = conf or {}
end

function CartoonDataLoader:name()
  return 'CartoonDataLoader'
end

function CartoonDataLoader:Initialize(opt)
  opt.align_data = 0
  self.dataA = data_util.load_dataset('A', opt, opt.input_nc)
  -- print(opt.output_nc)
  self.dataB = data_util.load_dataset('B', opt, opt.output_nc)
  self.dataB_edge = data_util.load_dataset('B_edge', opt, opt.output_nc)
end

-- actually fetches the data
-- |return|: a table of two tables, each corresponding to
-- the batch for dataset A and dataset B
function CartoonDataLoader:LoadBatchForAllDatasets()
  local batchA, pathA = self.dataA:getBatch()
  local batchB, pathB = self.dataB:getBatch()
  local batchB_edge, pathB_edge = self.dataB_edge:getBatch()
  return batchA, batchB, batchB_edge, pathA, pathB, pathB_edge
end

-- returns the size of each dataset
function CartoonDataLoader:size(dataset)
  if dataset == 'B' then
    return self.dataB:size()
  end

  -- return the size of the first dataset by default
  return self.dataA:size()
end
