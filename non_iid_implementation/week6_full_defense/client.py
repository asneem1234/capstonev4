# Client with client-side fingerprint computation
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from config import Config
from attack import LabelFlippingAttack
from pq_crypto import PQCrypto, hash_update
from defense_fingerprint_client import ClientSideFingerprintDefense

class Client:
    def __init__(self, client_id, data_loader):
        self.client_id = client_id
        self.data_loader = data_loader
        self.is_malicious = client_id in Config.MALICIOUS_CLIENTS
        
        # Initialize attack if malicious
        if self.is_malicious and Config.ATTACK_ENABLED:
            self.attack = LabelFlippingAttack(num_classes=10)
        else:
            self.attack = None
        
        # Initialize PQ crypto
        if Config.USE_PQ_CRYPTO:
            self.pq_crypto = PQCrypto(use_real_crypto=Config.USE_REAL_CRYPTO)
            self.sig_public_key, self.sig_secret_key = self.pq_crypto.generate_sig_keypair()
        else:
            self.pq_crypto = None
            self.sig_public_key = None
            self.sig_secret_key = None
        
        # Initialize fingerprint defense (client-side computation)
        if Config.USE_FINGERPRINTS:
            self.fingerprint_defense = ClientSideFingerprintDefense(
                fingerprint_dim=Config.FINGERPRINT_DIM
            )
        else:
            self.fingerprint_defense = None
    
    def train(self, global_model, server_public_key=None, round_num=0):
        """Train on local data, return update with fingerprint and optional crypto"""
        # Copy global model
        model = copy.deepcopy(global_model)
        model.train()
        
        optimizer = optim.SGD(
            model.parameters(), 
            lr=Config.LEARNING_RATE
        )
        
        # Local training
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(Config.LOCAL_EPOCHS):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                # Apply attack if malicious
                if self.attack is not None:
                    data, target = self.attack.apply(data, target)
                
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += len(target)
        
        # Compute update: Δw = w_local - w_global
        update = {}
        update_norm = 0.0
        for name, param in model.named_parameters():
            global_param = dict(global_model.named_parameters())[name]
            delta = param.data - global_param.data
            update[name] = delta
            update_norm += torch.norm(delta).item() ** 2
        
        update_norm = update_norm ** 0.5
        train_acc = 100. * total_correct / total_samples
        avg_loss = total_loss / total_samples
        
        # CLIENT-SIDE: Compute fingerprint of update with metadata
        fingerprint = None
        if self.fingerprint_defense:
            fingerprint = self.fingerprint_defense.compute_fingerprint(
                update,
                train_loss=avg_loss,
                train_acc=train_acc
            )
        
        # Apply PQ crypto if enabled
        if Config.USE_PQ_CRYPTO and self.pq_crypto and server_public_key:
            # Sign the update (include fingerprint in signature for integrity)
            message = hash_update(update, fingerprint, self.client_id, round_num)
            signature = self.pq_crypto.sign_message(message, self.sig_secret_key)
            
            # Encrypt the update
            ciphertext, encrypted_data = self.pq_crypto.encrypt_update(update, server_public_key)
            
            return {
                'ciphertext': ciphertext,
                'encrypted_data': encrypted_data,
                'fingerprint': fingerprint,  # Sent in plaintext for clustering
                'signature': signature,
                'train_loss': avg_loss,  # Metadata for enhanced clustering
                'train_acc': train_acc,  # Metadata for enhanced clustering
                'client_id': self.client_id,
                'round_num': round_num
            }, train_acc, avg_loss, update_norm
        else:
            # No crypto - return plain update with fingerprint
            return {
                'update': update,
                'fingerprint': fingerprint,
                'client_id': self.client_id
            }, train_acc, avg_loss, update_norm
