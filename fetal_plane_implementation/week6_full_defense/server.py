# Server with full defense (fingerprint + validation + PQ crypto)
import torch
import torch.nn.functional as F
import numpy as np
from config import Config
from defense_validation import ValidationDefense
from defense_fingerprint_client import ClientSideFingerprintDefense
from pq_crypto import PQCrypto

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
        
        # Initialize defenses
        if Config.DEFENSE_ENABLED:
            self.validation_defense = ValidationDefense(
                validation_loader, 
                threshold=Config.VALIDATION_THRESHOLD
            )
            if Config.USE_FINGERPRINTS:
                self.fingerprint_defense = ClientSideFingerprintDefense(
                    fingerprint_dim=Config.FINGERPRINT_DIM
                )
            else:
                self.fingerprint_defense = None
        else:
            self.validation_defense = None
            self.fingerprint_defense = None
    
    def aggregate(self, client_messages, client_public_keys=None):
        """FedAvg with full defense stack"""
        
        # Step 1: Decrypt updates if PQ crypto enabled
        client_updates = []
        client_fingerprints = []
        client_losses = []
        client_accs = []
        
        if Config.USE_PQ_CRYPTO and self.pq_crypto:
            for msg in client_messages:
                try:
                    update = self.pq_crypto.decrypt_update(
                        msg['ciphertext'], 
                        msg['encrypted_data'], 
                        self.kem_secret_key
                    )
                    
                    # Verify fingerprint integrity
                    if self.fingerprint_defense and msg['fingerprint'] is not None:
                        is_valid, _, similarity = self.fingerprint_defense.verify_fingerprint(
                            update, msg['fingerprint']
                        )
                        if not is_valid:
                            print(f"  WARNING: Client fingerprint mismatch (sim={similarity:.4f})")
                            continue
                    
                    client_updates.append(update)
                    client_fingerprints.append(msg['fingerprint'])
                    client_losses.append(msg.get('train_loss', 0))
                    client_accs.append(msg.get('train_acc', 0))
                except Exception as e:
                    print(f"  ERROR: Failed to decrypt: {e}")
        else:
            # No crypto - extract plain updates
            for msg in client_messages:
                client_updates.append(msg['update'])
                client_fingerprints.append(msg.get('fingerprint'))
                client_losses.append(msg.get('train_loss', 0))
                client_accs.append(msg.get('train_acc', 0))
        
        if len(client_updates) == 0:
            print("  WARNING: No valid updates!")
            return 0.0, {}, {}, {}
        
        # Step 2: Defense filtering
        if self.validation_defense:
            # Fingerprint clustering
            honest_indices = []
            malicious_indices = []
            
            if self.fingerprint_defense and client_fingerprints[0] is not None:
                honest_indices, malicious_indices, _ = \
                    self.fingerprint_defense.cluster_fingerprints(
                        client_fingerprints,
                        train_losses=client_losses if Config.USE_METADATA_FEATURES else None,
                        train_accs=client_accs if Config.USE_METADATA_FEATURES else None,
                        cosine_threshold=Config.COSINE_THRESHOLD
                    )
                
                print(f"  Fingerprint clustering: {len(honest_indices)} honest, "
                      f"{len(malicious_indices)} suspicious")
            
            # Validation filtering (only on suspicious)
            filtered_updates = []
            for i, update in enumerate(client_updates):
                if i in honest_indices:
                    # Auto-accept honest cluster
                    filtered_updates.append(update)
                else:
                    # Validate suspicious
                    is_valid, _, _, loss_inc = \
                        self.validation_defense.validate_update(self.model, update)
                    if is_valid:
                        filtered_updates.append(update)
                    else:
                        print(f"  Rejected client {i} (loss increase={loss_inc:.4f})")
            
            updates_to_aggregate = filtered_updates
            print(f"  After defense: {len(updates_to_aggregate)}/{len(client_updates)} updates accepted")
        else:
            updates_to_aggregate = client_updates
        
        # Step 3: Aggregate (FedAvg)
        if len(updates_to_aggregate) == 0:
            print("  WARNING: No valid updates after filtering!")
            return 0.0, {}, {}, {}
        
        aggregated_update = {}
        for name in updates_to_aggregate[0].keys():
            aggregated_update[name] = torch.zeros_like(updates_to_aggregate[0][name])
        
        num_valid = len(updates_to_aggregate)
        for update in updates_to_aggregate:
            for name in update.keys():
                aggregated_update[name] += update[name] / num_valid
        
        # Compute norm
        agg_norm = sum(torch.norm(v).item() ** 2 for v in aggregated_update.values()) ** 0.5
        
        # Apply to global model
        for name, param in self.model.named_parameters():
            param.data += aggregated_update[name]
        
        defense_stats = {
            'total_updates': len(client_updates),
            'accepted_updates': len(updates_to_aggregate),
            'rejected_updates': len(client_updates) - len(updates_to_aggregate),
            'honest_indices': honest_indices if self.fingerprint_defense else [],
            'malicious_indices': malicious_indices if self.fingerprint_defense else []
        }
        
        return agg_norm, defense_stats, {}, {}
    
    def evaluate(self):
        """Evaluate global model on test set"""
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
