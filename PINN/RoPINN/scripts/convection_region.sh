python convection_region_optimization.py --model PINN --device 'cuda:0'
python convection_region_optimization.py --model QRes --device 'cuda:0'
python convection_region_optimization.py --model FLS --device 'cuda:0'
python convection_region_optimization.py --model KAN --device 'cuda:0' # for KAN, the best past_iterations is 15
python convection_region_optimization.py --model PINNsFormer --device 'cuda:0'