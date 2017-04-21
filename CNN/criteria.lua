--
-- User: peyman
-- Date: 11/2/16
-- Time: 10:23 AM
--


--[[
criteria

semantics of weights: a way of managing unequal number of classes in training. e.g
we have 1MM Negative (not leaf) examples and only 100 positive (leaf). Weights allow
us to express our preferences

"loss in misclassifying Negative class is 0.3"
"loss in misclassifying Positive class is 0.7"

Alternative way of handling this problem is use patch-based approach
--]]

if opt.loss == 'BCE' then
    loss = nn.BCECriterion()

elseif opt.loss == 'NLL' then
    loss = nn.ClassNLLCriterion(torch.Tensor{0.3, 0.7})
end

if opt.gpu then
    loss:cuda()
end
