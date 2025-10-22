# Simple Post-Quantum Cryptography wrapper
# Uses liboqs for Kyber (encryption) and Dilithium (signatures)

import pickle
import hashlib

# Check if liboqs is available
try:
    import oqs
    PQ_AVAILABLE = True
except ImportError:
    PQ_AVAILABLE = False
    print("INFO: liboqs not installed. PQ crypto will be simulated.")
    print("      Install liboqs-python for real post-quantum security.")

class PQCrypto:
    """Post-Quantum Cryptography for Federated Learning"""
    
    def __init__(self, use_real_crypto=False):
        """
        Args:
            use_real_crypto: If True, use real PQ crypto (requires liboqs).
                           If False, simulate for testing (no actual security).
        """
        self.use_real_crypto = use_real_crypto and PQ_AVAILABLE
        
        if self.use_real_crypto:
            # Use real liboqs
            self.kem = oqs.KeyEncapsulation("Kyber512")
            self.sig = oqs.Signature("Dilithium2")
        else:
            # Simulation mode (for testing without liboqs)
            self.kem = None
            self.sig = None
    
    def generate_kem_keypair(self):
        """Generate Kyber512 keypair for encryption"""
        if self.use_real_crypto:
            public_key = self.kem.generate_keypair()
            secret_key = self.kem.export_secret_key()
            return public_key, secret_key
        else:
            # Simulated keys (not secure!)
            return b"simulated_public_key", b"simulated_secret_key"
    
    def generate_sig_keypair(self):
        """Generate Dilithium2 keypair for signatures"""
        if self.use_real_crypto:
            public_key = self.sig.generate_keypair()
            secret_key = self.sig.export_secret_key()
            return public_key, secret_key
        else:
            # Simulated keys (not secure!)
            return b"simulated_sig_public", b"simulated_sig_secret"
    
    def encrypt_update(self, update, public_key):
        """
        Encrypt model update using Kyber512
        
        Args:
            update: Dict of tensors
            public_key: Server's public key
        
        Returns:
            (ciphertext, encrypted_data)
        """
        # Serialize update
        serialized = pickle.dumps(update)
        
        if self.use_real_crypto:
            # Real Kyber encryption
            ciphertext, shared_secret = self.kem.encap_secret(public_key)
            
            # Use shared secret as encryption key (simplified)
            encrypted_data = self._xor_encrypt(serialized, shared_secret)
            
            return ciphertext, encrypted_data
        else:
            # Simulated encryption (just return plaintext for testing)
            return b"simulated_ciphertext", serialized
    
    def decrypt_update(self, ciphertext, encrypted_data, secret_key):
        """
        Decrypt model update using Kyber512
        
        Args:
            ciphertext: Encapsulated key
            encrypted_data: Encrypted update
            secret_key: Server's secret key
        
        Returns:
            update: Dict of tensors
        """
        if self.use_real_crypto:
            # Real Kyber decryption
            shared_secret = self.kem.decap_secret(ciphertext)
            
            # Decrypt data
            serialized = self._xor_encrypt(encrypted_data, shared_secret)  # XOR is symmetric
            
            return pickle.loads(serialized)
        else:
            # Simulated decryption
            return pickle.loads(encrypted_data)
    
    def sign_message(self, message, secret_key):
        """Sign a message using Dilithium2"""
        if self.use_real_crypto:
            if isinstance(message, str):
                message = message.encode()
            return self.sig.sign(message)
        else:
            # Simulated signature
            return b"simulated_signature"
    
    def verify_signature(self, message, signature, public_key):
        """Verify a signature using Dilithium2"""
        if self.use_real_crypto:
            if isinstance(message, str):
                message = message.encode()
            return self.sig.verify(message, signature, public_key)
        else:
            # Simulated verification (always succeeds)
            return True
    
    def _xor_encrypt(self, data, key):
        """Simple XOR encryption (for demonstration)"""
        # Expand key to match data length
        key_expanded = (key * (len(data) // len(key) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_expanded))

def hash_update(update, fingerprint=None, client_id=None, round_num=None):
    """
    Create a hash of the update for signing
    
    Args:
        update: Dict of tensors
        fingerprint: Optional fingerprint vector
        client_id: Optional client ID
        round_num: Optional round number
        
    Returns:
        Hash string
    """
    # Serialize update
    serialized = pickle.dumps(update)
    
    # Add fingerprint if provided
    if fingerprint is not None:
        serialized += pickle.dumps(fingerprint)
    
    # Add metadata
    if client_id is not None:
        serialized += str(client_id).encode()
    if round_num is not None:
        serialized += str(round_num).encode()
    
    # Compute hash
    return hashlib.sha256(serialized).hexdigest()
