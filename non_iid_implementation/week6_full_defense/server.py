# Server with client-side fingerprint verification
import torch
import torch.nn.functional as F
from config import Config
from defense_validation import ValidationDefense
from defense_fingerprint_client import ClientSideFingerprintDefense
from defense_adaptive import AdaptiveDefense
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
            self.adaptive_defense = None  # DISABLED - using simple norm filter
            if Config.USE_FINGERPRINTS:
                self.fingerprint_defense = ClientSideFingerprintDefense(
                    fingerprint_dim=Config.FINGERPRINT_DIM
                )
            else:
                self.fingerprint_defense = None
        else:
            self.validation_defense = None
            self.adaptive_defense = None
            self.fingerprint_defense = None
    
    def aggregate(self, client_messages, client_public_keys):
        """FedAvg with PQ crypto + client-side fingerprint verification + validation"""
        validation_results = []
        fingerprint_results = None
        crypto_results = {'verified': 0, 'failed': 0, 'decrypted': 0}
        integrity_results = {'verified': 0, 'failed': 0, 'tampered': []}
        
        # Step 1: Verify and decrypt if PQ crypto is enabled
        client_updates = []
        client_fingerprints = []
        
        if Config.USE_PQ_CRYPTO and self.pq_crypto:
            for i, msg in enumerate(client_messages):
                try:
                    # Decrypt update
                    update = self.pq_crypto.decrypt_update(
                        msg['ciphertext'], 
                        msg['encrypted_data'], 
                        self.kem_secret_key
                    )
                    
                    # SERVER-SIDE: Verify fingerprint integrity
                    if self.fingerprint_defense and msg['fingerprint'] is not None:
                        is_valid, actual_fp, similarity = self.fingerprint_defense.verify_fingerprint(
                            update, msg['fingerprint']
                        )
                        
                        if is_valid:
                            integrity_results['verified'] += 1
                        else:
                            integrity_results['failed'] += 1
                            integrity_results['tampered'].append(i)
                            print(f"  WARNING: Client {i} fingerprint mismatch! "
                                  f"Similarity={similarity:.4f} (expected ~1.0)")
                            continue  # Reject tampered update
                    
                    client_updates.append(update)
                    client_fingerprints.append(msg['fingerprint'])
                    crypto_results['decrypted'] += 1
                    crypto_results['verified'] += 1
                    
                except Exception as e:
                    print(f"  WARNING: Failed to process client {i}: {e}")
                    crypto_results['failed'] += 1
        else:
            # No crypto - extract plain updates
            for i, msg in enumerate(client_messages):
                client_updates.append(msg['update'])
                client_fingerprints.append(msg['fingerprint'])
        
        if len(client_updates) == 0:
            print("  WARNING: No valid updates after crypto/integrity verification!")
            return 0.0, validation_results, fingerprint_results, crypto_results, integrity_results
        
        # Step 2: ADAPTIVE Byzantine defense
        if False and self.adaptive_defense:  # DISABLED - using simple norm filter instead
            # Collect metadata
            metadata = None
            if len(client_messages) > 0 and 'train_loss' in client_messages[0]:
                losses = [msg['train_loss'] for msg in client_messages]
                accuracies = [msg['train_acc'] for msg in client_messages]
                metadata = {'losses': losses, 'accuracies': accuracies}
            
            # Extract features from all updates
            features = self.adaptive_defense.compute_update_features(
                client_updates, 
                self.model,
                client_metadata=metadata
            )
            
            # Adaptive anomaly detection (NO hard-coded thresholds!)
            honest_indices, malicious_indices, separation_factor, diagnostics = \
                self.adaptive_defense.detect_malicious_adaptive(
                    features, 
                    method='statistical'  # Options: 'statistical', 'clustering', 'dbscan', 'isolation_forest'
                )
            
            # ADAPTIVE results replace all previous filtering
            fingerprint_results = {
                'main_cluster': honest_indices,
                'outliers': malicious_indices,
                'method': 'adaptive_' + diagnostics['method'],
                'separation_factor': separation_factor,
                'diagnostics': diagnostics
            }
            
            # OLD FINGERPRINT CLUSTERING (kept for comparison, but not used)
            old_fingerprint_results = None
            if self.fingerprint_defense and client_fingerprints[0] is not None:
                # Collect metadata (loss/accuracy) from messages
                metadata = None
                if Config.USE_METADATA_FEATURES:
                    losses = []
                    accuracies = []
                    for msg in client_messages:
                        if 'train_loss' in msg and 'train_acc' in msg:
                            losses.append(msg['train_loss'])
                            accuracies.append(msg['train_acc'])
                    
                    if len(losses) == len(client_fingerprints):
                        metadata = {'losses': losses, 'accuracies': accuracies}
                
                main_cluster, outliers = self.fingerprint_defense.cluster_fingerprints(
                    client_fingerprints,
                    threshold=Config.COSINE_THRESHOLD,
                    metadata=metadata
                )
                old_fingerprint_results = {
                    'main_cluster': main_cluster,
                    'outliers': outliers,
                    'method': 'client-side + metadata (old)'
                }
            
            # ADAPTIVE FILTERING: Only aggregate honest updates
            filtered_updates = []
            for i in honest_indices:
                filtered_updates.append(client_updates[i])
                validation_results.append({
                    'client_id': i,
                    'valid': True,
                    'method': 'adaptive_honest',
                    'loss_before': None,
                    'loss_after': None,
                    'loss_increase': features[i, 1]  # Loss increase from feature extraction
                })
            
            # Rejected updates (for logging)
            for i in malicious_indices:
                validation_results.append({
                    'client_id': i,
                    'valid': False,
                    'method': 'adaptive_malicious',
                    'loss_before': None,
                    'loss_after': None,
                    'loss_increase': features[i, 1]  # Loss increase from feature extraction
                })
            
            updates_to_aggregate = filtered_updates
        else:
            # SIMPLE NORM FILTERING + VALIDATION (proven approach)
            # Extract update norms
            update_norms = [msg['update_norm'] for msg in client_messages]
            
            # Calculate median and threshold
            median_norm = torch.tensor(update_norms).median().item()
            norm_threshold = median_norm * 3.0
            
            print(f"\n  [NORM FILTERING]")
            print(f"    Median norm: {median_norm:.4f}")
            print(f"    Threshold (3×median): {norm_threshold:.4f}")
            
            # Flag suspicious clients
            suspicious_clients = []
            for i in range(len(client_updates)):
                if update_norms[i] > norm_threshold:
                    suspicious_clients.append(i)
                    print(f"    Client {i}: norm={update_norms[i]:.2f} > {norm_threshold:.2f} → SUSPICIOUS")
            
            # PURE NORM FILTERING: Reject high-norm, accept low-norm (no validation)
            filtered_updates = []
            for i in range(len(client_updates)):
                if i in suspicious_clients:
                    # REJECT high-norm clients (gradient ascent attack)
                    validation_results.append({
                        'client_id': i,
                        'valid': False,
                        'method': 'norm_filter_reject',
                        'loss_before': None,
                        'loss_after': None,
                        'loss_increase': None,
                        'suspicious': True
                    })
                else:
                    # ACCEPT low-norm clients (honest)
                    filtered_updates.append(client_updates[i])
                    validation_results.append({
                        'client_id': i,
                        'valid': True,
                        'method': 'norm_filter_accept',
                        'loss_before': None,
                        'loss_after': None,
                        'loss_increase': None,
                        'suspicious': False
                    })
            
            updates_to_aggregate = filtered_updates
        
        # Check if we have any valid updates
        if len(updates_to_aggregate) == 0:
            print("  WARNING: No valid updates! Skipping aggregation.")
            return 0.0, validation_results, fingerprint_results, crypto_results, integrity_results
        
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
        
        return agg_norm, validation_results, fingerprint_results, crypto_results, integrity_results
    
    def evaluate(self):
        """Evaluate global model on test set"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        
        test_loss /= total
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
