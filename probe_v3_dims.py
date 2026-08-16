import torch, math
from ultralytics import YOLO

m = YOLO('YOLOv3/yolov3u.pt').to('cuda')
model = m.model

# Find classification layer
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        w = module.weight
        if w.ndim == 4 and w.shape[0] == 80 and w.shape[1] < 1000:
            D = w.shape[1]
            w_flat = w.mean(dim=(2,3)).detach()  # (80, D)
            w_person = w_flat[0]
            w_meter = w_flat[12]
            
            print(f'Layer: {name}')
            print(f'D = {D}, n = {math.sqrt(D):.1f}')
            print(f'cos_sim(person, meter) = {torch.nn.functional.cosine_similarity(w_person.unsqueeze(0), w_meter.unsqueeze(0)).item():.4f}')
            print(f'||w_person - w_meter||_2 = {torch.norm(w_person - w_meter).item():.4f}')
            print()
            
            # Classify dims by contribution threshold (10% of max)
            p_thresh = w_person.abs().max() * 0.1
            m_thresh = w_meter.abs().max() * 0.1
            
            person_only = (w_person.abs() > p_thresh) & (w_meter.abs() < m_thresh)
            meter_only = (w_meter.abs() > m_thresh) & (w_person.abs() < p_thresh)
            shared = (w_person.abs() > p_thresh) & (w_meter.abs() > m_thresh)
            inactive = (~person_only) & (~meter_only) & (~shared)
            
            p_idx = person_only.nonzero(as_tuple=True)[0].tolist()
            m_idx = meter_only.nonzero(as_tuple=True)[0].tolist()
            s_idx = shared.nonzero(as_tuple=True)[0].tolist()
            i_count = inactive.sum().item()
            
            print(f'Person-only dims ({len(p_idx)}): {p_idx}')
            print(f'Meter-only dims ({len(m_idx)}): {m_idx}')
            print(f'Shared dims ({len(s_idx)}): {s_idx}')
            print(f'Inactive dims: {i_count}/{D}')
            print()
            
            # Show actual weight values for person-only and meter-only
            print('Person-only dim weights (w_person value):')
            for d in p_idx:
                print(f'  dim {d}: w_person={w_person[d].item():.4f}, w_meter={w_meter[d].item():.4f}')
            print()
            print('Meter-only dim weights:')
            for d in m_idx:
                print(f'  dim {d}: w_person={w_person[d].item():.4f}, w_meter={w_meter[d].item():.4f}')
            print()
            print('Shared dim weights:')
            for d in s_idx:
                print(f'  dim {d}: w_person={w_person[d].item():.4f}, w_meter={w_meter[d].item():.4f}')
            break

del m
torch.cuda.empty_cache()
