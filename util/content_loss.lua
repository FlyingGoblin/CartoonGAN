require 'torch'
require 'nn'
local content = {}

function content.defineVGG(content_layer)
  local contentFunc = nn.Sequential()
  --require 'loadcaffe'
  require 'util/VGG_preprocess'
  --cnn = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'cudnn')
  require 'cudnn'
  cnn = torch.load('./VGG_ILSVRC_19_layers.t7')

  contentFunc:add(nn.SpatialUpSamplingBilinear({oheight=224, owidth=224}))
  contentFunc:add(nn.VGG_postprocess())
  for i = 1, #cnn do
    local layer = cnn:get(i):clone()
    local name = layer.name
    local layer_type = torch.type(layer)
    contentFunc:add(layer)
    if name == content_layer then
      print("Setting up content layer: ", layer.name)
      break
    end
  end
  cnn = nil
  collectgarbage()
  print(contentFunc)
  return contentFunc
end

function content.defineAlexNet(content_layer)
  local contentFunc = nn.Sequential()
  require 'loadcaffe'
  require 'util/VGG_preprocess'
  cnn = loadcaffe.load('../models/alexnet.prototxt', '../models/alexnet.caffemodel', 'cudnn')
  contentFunc:add(nn.SpatialUpSamplingBilinear({oheight=224, owidth=224}))
  contentFunc:add(nn.VGG_postprocess())
  for i = 1, #cnn do
    local layer = cnn:get(i):clone()
    local name = layer.name
    local layer_type = torch.type(layer)
    contentFunc:add(layer)
    if name == content_layer then
      print("Setting up content layer: ", layer.name)
      break
    end
  end
  cnn = nil
  collectgarbage()
  print(contentFunc)
  return contentFunc
end


function content.defineHED()
  local hed = nn.Sequential()
  require 'loadcaffe'
  -- require 'caffegraph'
  require 'util/VGG_preprocess'
  cnn = loadcaffe.load('../models/hed.prototxt', '../models/hed.caffemodel', 'cudnn')
  hed:add(nn.SpatialUpSamplingBilinear({oheight=500, owidth=500}))
  hed:add(nn.VGG_postprocess())
  hed:add(cnn)
  return hed
end


function content.defineVGGClf()
  local clf = nn.Sequential()
  require 'loadcaffe'
  require 'util/VGG_preprocess'
  cnn = loadcaffe.load('VGG_ILSVRC_19_layers.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'cudnn')
  clf:add(nn.SpatialUpSamplingBilinear({oheight=224, owidth=224}))
  clf:add(nn.VGG_postprocess())
  clf:add(cnn)
  return clf
end

function content.defineContent(content_loss, layer_name)
  -- print('content_loss_define', content_loss)
  if content_loss == 'pixel' or content_loss == 'none' then
    return nil
  elseif content_loss == 'vgg' then
    return content.defineVGG(layer_name)
  else
    print("unsupported content loss")
    return nil
  end
end


function content.lossUpdate(criterionContent, real_source, fake_target, contentFunc, loss_type, weight)
  if loss_type == 'none' then
    local errCont = 0.0
    local df_d_content = torch.zeros(fake_target:size())
    return errCont, df_d_content
  elseif loss_type == 'pixel' then
    local errCont = criterionContent:forward(fake_target, real_source) * weight
    local df_do_content = criterionContent:backward(fake_target, real_source)*weight
    print(df_do_content:sum())
    --os.exit()
    return errCont, df_do_content
  elseif loss_type == 'vgg' then
    local f_real = contentFunc:forward(real_source):clone()
    local f_fake = contentFunc:forward(fake_target):clone()
    local errCont = criterionContent:forward(f_fake, f_real) * weight
    local df_do_tmp = criterionContent:backward(f_fake, f_real) * weight
    --print(f_fake:size())
    --print(df_do_tmp:size())
    local df_do_content = contentFunc:updateGradInput(fake_target, df_do_tmp)--:mul(weight)
    --print(df_do_content:size())
    --os.exit()
    --print(df_do_content:sum())
    --os.exit()
    --print(df_do_content:sub(1,1,2,2,3,3))
    --os.exit()
    return errCont, df_do_content
  else error("unsupported content loss")
  end
end


return content
