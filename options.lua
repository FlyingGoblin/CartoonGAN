--------------------------------------------------------------------------------
-- Configure options
--------------------------------------------------------------------------------

local options = {}
-- options for train
local opt_train = {
   DATA_ROOT = '',         -- path to images (should have subfolders 'train_A', 'train_B', 'train_B_edge')
   batchSize = 8,          -- # images in batch
   loadSize = 256,         -- scale images to this size
   fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 32,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   niter = 100,            -- #  of iter at starting learning rate
   niter_decay = 100,      --  # of iter to linearly decay learning rate to zero
   lr = 0.00001,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   display_id = 10,        -- display window id.
   display_winsize = 128,  -- display window size
   display_freq = 25,      -- display the current results every display_freq iterations
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = '',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'AtoB',    -- AtoB or BtoA for CycleGAN
   phase = 'train',             -- train, val, test, etc
   nThreads = 1,                -- # threads for loading data
   save_epoch_freq = 5,         -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 100,             -- print the debug information every print_freq iterations
   save_display_freq = 500,    -- save the current display of results every save_display_freq_iterations
   continue_train = 0,          -- if continue training, load the latest model: 1: true, 0: false
   which_epoch = 'latest',            -- which epoch to continue train? set to 'latest' to use latest cached model
   init_model = '',            -- path to init model. if set to '' training will start with init process
   init_lr = 0.0002,            -- initial learning rate for adam in init process
   init_niter = 10,             -- #  of iter at starting learning rate in init process
   init_niter_decay = 10,       -- #  of iter to linearly decay learning rate to zero in init process
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   checkpoints_dir = './checkpoints', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   which_model_netD = 'n_layers_D',        -- selects model to use for netD
   which_model_netG = 'resnet_nblocks',   -- selects model to use for netG
   norm = 'instance',             -- batch or instance normalization
   n_layers_D = 3,                -- n in which_model_netD=='n_layers'
   n_block_G = 16,                -- n in which_model_netG = 'resnet_nblocks'
   content_loss = 'vgg',        -- content loss type: L1, conv-layer, edge
   layer_name = 'conv4_4',          -- layer used in content loss
   lambda_A = 10.0,               -- weight for contant loss & weight for cycle loss (A -> B -> A) cycle_gan
   lambda_B = 10.0,               -- weight for cycle loss (B -> A -> B)
   model = 'cartoon_gan',           -- which mode to run. 'cartoon_gan', 'cycle_gan', 'pix2pix', 'bigan', 'content_gan'
   use_lsgan = 1,                 -- if 1, use least square GAN, if 0, use vanilla GAN
   align_data = 1,                -- if = 1, use dataloader for cartoon. if > 1, use the dataloader for where the images are aligned
   pool_size = 50,                -- the size of image buffer that stores previously generated images
   resize_or_crop='resize_and_crop',  -- resizing/cropping strategy
   identity = 0,                  -- use identity mapping for CycleGAN. Setting opt.identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set opt.identity = 0.1
}

-- options for test
opt_test = {
  DATA_ROOT = '',           -- path to images (should have subfolders 'test_A'/'test_B', etc)
  loadSize = 128,           -- scale images to this size
  fineSize = 128,           --  then crop to this size
  flip = 0,                  -- horizontal mirroring data augmentation
  display = 1,              -- display samples while training. 0 = false
  display_id = 200,         -- display window id.
  gpu = 1,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  how_many = 'all',         -- how many test images to run (set to all to run on every image found in the data/phase folder)
  phase = 'test',            -- train, val, test, etc
  preprocess = 'regular',   -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
  aspect_ratio = 1.0,       -- aspect ratio of result images
  norm = 'instance',        -- batchnorm or isntance norm
  name = '',                -- name of experiment, selects which model to run, should generally should be passed on command line
  input_nc = 3,              -- #  of input image channels
  output_nc = 3,             -- #  of output image channels
  serial_batches = 1,        -- if 1, takes images in order to make batches, otherwise takes them randomly
  cudnn = 1,                 -- set to 0 to not use cudnn (untested)
  checkpoints_dir = './checkpoints', -- loads models from here
  results_dir='./results/',          -- saves results here
  which_epoch = 'latest',            -- which epoch to test? set to 'latest' to use latest cached model
  model = 'cycle_gan',               -- which mode to run. 'cycle_gan', 'pix2pix', 'bigan', 'content_gan'; to use pretrained model, select `one_direction_test`
  align_data = 0,                    -- if > 0, use the dataloader for pix2pix
  which_direction = 'AtoB',          -- AtoB or BtoA
  resize_or_crop = 'resize_and_crop',  -- resizing/cropping strategy
}

--------------------------------------------------------------------------------
-- util functions
--------------------------------------------------------------------------------
function options.clone(opt)
  local copy = {}
  for orig_key, orig_value in pairs(opt) do
    copy[orig_key] = orig_value
  end
  return copy
end

function options.parse_options(mode)
  if mode == 'train' then
    opt = opt_train
    opt.test = 0
  elseif mode == 'test' then
    opt = opt_test
    opt.test = 1
  else
    print("Invalid option [" .. mode .. "]")
    return nil
  end

  -- one-line argument parser. parses enviroment variables to override the defaults
  for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
  if mode == 'test' then
    opt.nThreads = 1
    opt.continue_train = 1
    opt.batchSize = 1  -- test code only supports batchSize=1
  end

  -- print by keys
  keyset = {}
  for k,v in pairs(opt) do
    table.insert(keyset, k)
  end
  table.sort(keyset)
  print("------------------- Options -------------------")
  for i,k in ipairs(keyset) do
    print(("%+25s: %s"):format(k, opt[k]))
  end
  print("-----------------------------------------------")

  -- save opt to checkpoints
  paths.mkdir(opt.checkpoints_dir)
  paths.mkdir(paths.concat(opt.checkpoints_dir, opt.name))

  -- save opt to the disk
  fd = io.open(paths.concat(opt.checkpoints_dir, opt.name, 'opt_' .. mode .. '.txt'), 'w')
  for i,k in ipairs(keyset) do
    fd:write(("%+25s: %s\n"):format(k, opt[k]))
  end
  fd:close()

  return opt
end


return options
