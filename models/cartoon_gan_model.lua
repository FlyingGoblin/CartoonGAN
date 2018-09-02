local class = require 'class'
require 'models.base_model'
require 'models.architectures'
require 'util.image_pool'
util = paths.dofile('../util/util.lua')
content = paths.dofile('../util/content_loss.lua')

CartoonGANModel = class('CartoonGANModel', 'BaseModel')

function CartoonGANModel:__init(conf)
  BaseModel.__init(self, conf)
  conf = conf or {}
end

function CartoonGANModel:model_name()
  return 'CartoonGANModel'
end

function CartoonGANModel:InitializeStatesD()
  local optimState = {learningRate=opt.lr, beta1=opt.beta1,}
  return optimState
end

function CartoonGANModel:InitializeStatesG()
  local optimState = {learningRate=opt.lr, beta1=opt.beta1,}
  return optimState
end
-- Defines models and networks
function CartoonGANModel:Initialize(opt)
  if opt.test == 0 then
    self.fakePool = ImagePool(opt.pool_size)
  end
  -- define tensors
  self.real_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
  self.fake_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  self.real_B = self.fake_B:clone() --torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  self.edge_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

  -- load/define models
  self.criterionGAN = nn.MSECriterion()
  self.criterionContent = nn.AbsCriterion()
  --self.criterionContent = nn.MSECriterion()

  self.contentFunc = content.defineContent(opt.content_loss, opt.layer_name)
  self.netG, self.netD = nil, nil
  print('init:'..string.len(opt.init_model))
  if opt.continue_train == 1 then
    if opt.which_epoch then -- which_epoch option exists in test mode
      self.netG = util.load_test_model('G_A', opt)
      self.netD = util.load_test_model('D_A', opt)
    else
      self.netG = util.load_model('G_A', opt)
      self.netD = util.load_model('D_A', opt)
    end
  elseif string.len(opt.init_model) ~= 0 then
    self.netG = util.load_init_model(opt.init_model, opt)
    self.netD = defineD(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, false)
  else
    self.netG = defineG(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.n_block_G)
    self.netD = defineD(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, false)
  end
  print('netG...', self.netG)
  for i=1 , self.netG:size(1) do 
    print(string.format("%d:%s", i, self.netG.modules[i]))
  end
  print('netD...', self.netD)
  -- define real/fake labels
  netD_output_size = self.netD:forward(self.real_A):size()
  self.fake_label = torch.Tensor(netD_output_size):fill(0.0)
  self.real_label = torch.Tensor(netD_output_size):fill(1.0) -- no soft smoothing
  self.optimStateD = self:InitializeStatesD()
  self.optimStateG = self:InitializeStatesG()
  self:RefreshParameters()
  print('---------- # Learnable Parameters --------------')
  print(('G = %d'):format(self.parametersG:size(1)))
  print(('D = %d'):format(self.parametersD:size(1)))
  print('------------------------------------------------')
  self.lambda = opt.lambda_A
  self.errG = 0
  self.errD = 0
  -- init precess
  if opt.init_model == nil then
  end
end

function CartoonGANModel:fGx_init_basic(x, netG_source, netD_source, real_source, fake_target,
                                   gradParametersG_source, opt)
  util.BiasZero(netD_source)
  util.BiasZero(netG_source)
  gradParametersG_source:zero()
  local output = netD_source.output -- [hack] forward was already executed in fDx, so save computation netD_source:forward(fake_B) ---
  local errGAN = 0

  -- content loss
  local errContent, df_d_content = 0, 0
  errContent, df_d_content = content.lossUpdate(self.criterionContent, real_source, fake_target, self.contentFunc, opt.content_loss, self.lambda)
  errContent = errContent/self.lambda
  netG_source:forward(real_source)
  netG_source:backward(real_source, df_d_content)

  return gradParametersG_source, errGAN, errContent
end

function CartoonGANModel:fGx_init(x, opt)
  self.gradparametersG, self.errG, self.errCont =
  self:fGx_init_basic(x, self.netG, self.netD, self.real_A, self.fake_B,
             self.gradparametersG, opt)
  return self.errCont, self.gradparametersG
  --return self.errG, self.gradparametersG
end

function CartoonGANModel:OptimizeInitParameters(opt)
  local fGx = function(x) return self:fGx_init(x, opt) end
  optim.adam(fGx, self.parametersG, self.optimStateG)
end

-- Runs the forward pass of the network and
-- saves the result to member variables of the class
function CartoonGANModel:Forward(input, opt)
  if opt.which_direction == 'BtoA' then
    local temp = input.real_A
    input.real_A = input.real_B
    input.real_B = temp
  end

  self.real_A:copy(input.real_A)
  self.real_B:copy(input.real_B)
  self.edge_B:copy(input.edge_B)
  self.fake_B = self.netG:forward(self.real_A):clone()
  -- output = {self.fake_B}
  output =  {}
  -- if opt.test == 1 then

  -- end
  return output
end

-- create closure to evaluate f(X) and df/dX of discriminator
function CartoonGANModel:fDx_basic(x, gradParams, netD, netG,
                                   real_B, real_A, fake_B, edge_B, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  gradParams:zero()
  local errD_real, errD_rec, errD_fake, errD = 0, 0, 0, 0
  
  -- RealB  log(D_A(B))
  local output = netD:forward(real_B)
  errD_real_B = self.criterionGAN:forward(output, self.real_label)
  df_do_B = self.criterionGAN:backward(output, self.real_label)
  netD:backward(real_B, df_do_B)

  --EdgeB + log(1 - D_A(B'))
  output = self.netD:forward(edge_B)
  errD_edge_B = self.criterionGAN:forward(output, self.fake_label)
  df_do_B_edge = self.criterionGAN:backward(output, self.fake_label)
  self.netD:backward(edge_B, df_do_B_edge)

  -- Fake  + log(1 - D_A(G_A(A)))
  output = netD:forward(fake_B)
  errD_fake = self.criterionGAN:forward(output, self.fake_label)
  df_do = self.criterionGAN:backward(output, self.fake_label)
  netD:backward(fake_B, df_do)

  errD = (errD_real_B + errD_fake + errD_edge_B) / 3.0
  -- print('errD', errD
  return errD, gradParams
end


function CartoonGANModel:fDx(x, opt)
  fake_B = self.fakePool:Query(self.fake_B)
  self.errD, gradParams = self:fDx_basic(x, self.gradparametersD, self.netD, self.netG,
                                     self.real_B, self.real_A ,fake_B, self.edge_B, opt)
  return self.errD, gradParams
end

function CartoonGANModel:fGx_basic(x, netG_source, netD_source, real_source, real_target, fake_target,
                                   gradParametersG_source, opt)
  util.BiasZero(netD_source)
  util.BiasZero(netG_source)
  gradParametersG_source:zero()
  -- GAN loss
  local output = netD_source.output -- [hack] forward was already executed in fDx, so save computation netD_source:forward(fake_B) ---
  local errGAN = self.criterionGAN:forward(output, self.real_label)
  local df_do = self.criterionGAN:backward(output, self.real_label)
  local df_d_GAN = netD_source:updateGradInput(fake_target, df_do) ---:narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
  netG_source:forward(real_source)
  netG_source:backward(real_source, df_d_GAN)

  -- content loss
  local errContent, df_d_content = 0, 0
  errContent, df_d_content = content.lossUpdate(self.criterionContent, real_source, fake_target, self.contentFunc, opt.content_loss, self.lambda)
  errContent = errContent/self.lambda
  netG_source:forward(real_source)
  netG_source:backward(real_source, df_d_content)

  return gradParametersG_source, errGAN, errContent
end

function CartoonGANModel:fGx(x, opt)
  self.gradparametersG, self.errG, self.errCont =
  self:fGx_basic(x, self.netG, self.netD,
             self.real_A, self.real_B, self.fake_B,
             self.gradparametersG, opt)
  return self.errCont, self.gradparametersG
  --return self.errG, self.gradparametersG
end

function CartoonGANModel:OptimizeParameters(opt)
  local fDx = function(x) return self:fDx(x, opt) end
  local fGx = function(x) return self:fGx(x, opt) end
  optim.adam(fDx, self.parametersD, self.optimStateD)
  optim.adam(fGx, self.parametersG, self.optimStateG)
end

function CartoonGANModel:RefreshParameters()
  self.parametersD, self.gradparametersD = nil, nil -- nil them to avoid spiking memory
  self.parametersG, self.gradparametersG = nil, nil
  -- define parameters of optimization
  self.parametersG, self.gradparametersG = self.netG:getParameters()
  self.parametersD, self.gradparametersD = self.netD:getParameters()
end

function CartoonGANModel:Save(prefix, opt)
  util.save_model(self.netG, prefix .. '_net_G_A.t7', 1.0)
  util.save_model(self.netD, prefix .. '_net_D_A.t7', 1.0)
end

function CartoonGANModel:GetCurrentErrorDescription()
  description = ('G: %.4f  D: %.4f  Content: %.4f'):format(self.errG and self.errG or -1,
                         self.errD and self.errD or -1,
                         self.errCont and self.errCont or -1)
  return description
end


function CartoonGANModel:GetCurrentErrors()
  local errors = {errG=self.errG and self.errG or -1, errD=self.errD and self.errD or -1,
  errCont=self.errCont and self.errCont or -1}
  return errors
end

-- returns a string that describes the display plot configuration
function CartoonGANModel:DisplayPlot(opt)
  return 'errG,errD,errCont'
end


function CartoonGANModel:GetCurrentVisuals(opt, size)
  if not size then
    size = opt.display_winsize
  end
  --print(self.real_A:sub(1,1):size())
  --print(self.fake_B:sub(1,1):size())
  local real_A = self.real_A:sub(1,1):clone()
  local fake_B = self.fake_B:sub(1,1):clone()
  local real_B = self.real_B:sub(1,1):clone()

  local visuals = {}
  table.insert(visuals, {img=real_A, label='real_A'})
  table.insert(visuals, {img=fake_B, label='fake_B'})
  table.insert(visuals, {img=real_B, label='real_B'})
  return visuals
end

function CartoonGANModel:UpdateLearningRate(opt, init)
    local lrd, old_lr, lr = 0, 0
    if init then
        lrd = opt.init_lr / opt.init_niter_decay
        old_lr = self.optimStateG['learningRate']
        lr =  old_lr - lrd
    else
        lrd = opt.lr / opt.niter_decay
        old_lr = self.optimStateD['learningRate']
        lr =  old_lr - lrd
        self.optimStateD['learningRate'] = lr
    end
    self.optimStateG['learningRate'] = lr
    print(('update learning rate: %f -> %f'):format(old_lr, lr))
    --local llambda = (opt.lambda_A - 1) / opt.niter_decay
    --local old_lambda = self.lambda
    --self.lambda = old_lambda - llambda
    --print(('update lambda: %f -> %f'):format(old_lambda, self.lambda))
end
