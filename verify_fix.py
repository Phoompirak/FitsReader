
import spectral_fitting as sf
import numpy as np
import sys

def test_brute_force_updates():
    print("Testing brute_force_fit_parallel updates...")
    
    # Mock data
    energy = np.linspace(1, 10, 100)
    observed_rate = np.random.rand(100)
    observed_error = np.ones(100) * 0.1
    
    model_components = ['powerlaw']
    param_ranges = {
        'pl_norm': (0.1, 1.0),
        'photon_index': (1.0, 3.0)
    }
    
    # Run with small steps
    gen = sf.brute_force_fit_parallel(
        energy, observed_rate, observed_error,
        model_components, param_ranges,
        n_steps=3, n_workers=1,
        n_parts=1
    )
    
    updates_received = 0
    current_params_values = []
    
    for update in gen:
        updates_received += 1
        curr = update.get('current_params')
        best = update.get('best_params')
        curr_chi2 = update.get('current_chi2_dof')
        
        if update['status'] == 'running':
             if curr:
                 current_params_values.append(curr)
                 print(f"Update {updates_received}: Current={curr} | Chi2={curr_chi2} | Best={best}")
    
    if not current_params_values:
        print("❌ No current parameters received!")
        return False
        
    print(f"✅ Received {len(current_params_values)} updates with current parameters.")
    return True

if __name__ == "__main__":
    success = test_brute_force_updates()
    sys.exit(0 if success else 1)
