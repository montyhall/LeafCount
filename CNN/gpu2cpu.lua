--
-- User: peyman
-- Date: 11/21/16
-- Time: 9:02 AM
--

require 'paths'
require 'nn'
require 'cunn'
require 'netlighter'

----------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert a GPU model to a CPU one')
cmd:text('Example:')
cmd:text("th -i gpu2cpu.lua")
cmd:text('Options:')
-- Dataset options
cmd:option('-serializedModel', 'results/model.net', 'path to trained model')
cmd:option('-fname', 'results/model_CPU.net', 'name of the CPU model')
cmd:option('-save', 'results', 'save directory')
cmd:text()

opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

print(sys.COLORS.blue ..  '==> loading GPU built model')

torch.load(opt.serializedModel)
ok,model = pcall(torch.load, opt.serializedModel)
if not ok then
    print(sys.COLORS.red .. '... could not load model -> ',opt.serializedModel)
    return
end

print(sys.COLORS.blue ..  '==> converting to CPU model')

os.execute('mkdir -p ' .. sys.dirname(opt.fname))
print('==> saving CPU version of model to '..opt.fname)
model1 = model:clone()
model1 = model1:float()
netLighter(model1)
torch.save(opt.fname, model1)
print(sys.COLORS.blue ..  '==> DONE')
