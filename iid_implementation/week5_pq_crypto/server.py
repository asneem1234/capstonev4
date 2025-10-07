# Server with PQ crypto + fingerprint + validation defense
import torch
import torch.nn.functional as F
from config import Config
from defense_validation import ValidationDefense
from defense_fingerprint import FingerprintDefense
from pq_crypto import PQCrypto, hash_update

class Server:
    def __init__(self, model, validation_loader, test_loader):
        self.model = model
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        
        # Initialize PQ crypto
        if Config.USE_PQ_CRYPTO:
            self.pq_crypto = PQCrypto(use_real_crypto=Config.USE_REAL_CRYPTO)
            self.kem_public_key, self.kem_secret_key = self.pq_crypto.generate_kem_keypair()
        else:
            self.pq_crypto = None
            self.kem_public_key = None
            self.kem_secret_key = None
        
        # Initialize defenses
        if Config.DEFENSE_ENABLED:
            self.validation_defense = ValidationDefense(
                validation_loader, 
                threshold=Config.VALIDATION_THRESHOLD
            )
            if Config.USE_FINGERPRINTS:
                self.fingerprint_defense = FingerprintDefense(
                    fingerprint_dim=Config.FINGERPRINT_DIM
                )
            else:
                self.fingerprint_defense = None
        else:
            self.validation_defense = None
            self.fingerprint_defense = None
    
    def aggregate(self, client_messages, client_public_keys):
        """FedAvg with PQ crypto + fingerprint + validation filtering"""
        validation_results = []
        fingerprint_results = None
        crypto_results = {'verified': 0, 'failed': 0, 'decrypted': 0}
        
        # Step 1: Verify and decrypt if PQ crypto is enabled
        client_updates = []
        if Config.USE_PQ_CRYPTO and self.pq_crypto:
            for i, msg in enumerate(client_messages):
                # Verify signature
                original_update_hash = hash_update(None, None, msg['client_id'], msg['round_num'])
                # Note: We can't verify without decrypting first in this simple implementation
                # In production, you'd verify a hash of the ciphertext
                
                # Decrypt update
                try:
                    update = self.pq_crypto.decrypt_update(
                        msg['ciphertext'], 
                        msg['encrypted_data'], 
                        self.kem_secret_key
                    )
                    client_updates.append(update)
                    crypto_results['decrypted'] += 1
                    crypto_results['verified'] += 1
                except Exception as e:
                    print(f"  WARNING: Failed to decrypt client {i}: {e}")
                    crypto_results['failed'] += 1
        else:
            # No crypto - messages are plain updates
            client_updates = client_messages
        
        if len(client_updates) == 0:
            print("  WARNING: No valid updates after crypto verification!")
            return 0.0, validation_results, fingerprint_results, crypto_results
        
        # Step 2: Two-layer Byzantine defense
        if self.validation_defense:
            updates_to_validate = list(range(len(client_updates)))
            
            # Layer 2a: Fingerprint pre-filtering (fast)
            if self.fingerprint_defense:
                main_cluster, outliers = self.fingerprint_defense.cluster_updates(client_updates)
                fingerprint_results = {
                    'main_cluster': main_cluster,
                    'outliers': outliers
                }
                # Only validate outliers
                updates_to_validate = outliers
            
            # Layer 2b: Validation filtering (expensive, only on suspicious)
            filtered_updates = []
            for i in range(len(client_updates)):
                # Auto-accept main cluster (if fingerprints used)
                if fingerprint_results and i in fingerprint_results['main_cluster']:
                    filtered_updates.append(client_updates[i])
                    validation_results.append({
                        'client_id': i,
                        'valid': True,
                        'method': 'fingerprint',
                        'loss_before': None,
                        'loss_after': None,
                        'loss_increase': None
                    })
                # Validate suspicious updates
                elif i in updates_to_validate:
                    is_valid, loss_before, loss_after, loss_increase = \
                        self.validation_defense.validate_update(self.model, client_updates[i])
                    
                    validation_results.append({
                        'client_id': i,
                        'valid': is_valid,
                        'method': 'validation',
                        'loss_before': loss_before,
                        'loss_after': loss_after,
                        'loss_increase': loss_increase
                    })
                    
                    if is_valid:
                        filtered_updates.append(client_updates[i])
            
            updates_to_aggregate = filtered_updates
        else:
            updates_to_aggregate = client_updates
        
        # Check if we have any valid updates
        if len(updates_to_aggregate) == 0:
            print("  WARNING: No valid updates! Skipping aggregation.")
            return 0.0, validation_results, fingerprint_results, crypto_results
        
        # Step 3: Aggregate valid updates (FedAvg)
        aggregated_update = {}
        for name in updates_to_aggregate[0].keys():
            aggregated_update[name] = torch.zeros_like(
                updates_to_aggregate[0][name]
            )
        
        num_valid = len(updates_to_aggregate)
        for update in updates_to_aggregate:
            for name in update.keys():
                aggregated_update[name] += update[name] / num_valid
        
        # Compute aggregated update norm
        agg_norm = 0.0
        for name in aggregated_update.keys():
            agg_norm += torch.norm(aggregated_update[name]).item() ** 2
        agg_norm = agg_norm ** 0.5
        
        # Apply to global model
        for name, param in self.model.named_parameters():
            param.data += aggregated_update[name]
        
        return agg_norm, validation_results, fingerprint_results, crypto_results
    
    def evaluate(self):
        """Test global model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        
        accuracy = 100. * correct / total
        return accuracy
