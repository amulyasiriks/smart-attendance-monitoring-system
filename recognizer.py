# recognizer.py — ArcFace-based recognition (optional: insightface)
# If ArcFace is disabled or unavailable, the recognizer will return 'unknown'.

import os
# ArcFace (insightface) import is deferred and optional
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch

class Recognizer:
    def __init__(self, known_faces_dir, threshold=0.4, use_arcface=True):
        '''
        ArcFace recognizer using insightface's FaceAnalysis.
        known_faces_dir: folder with <id>_<FullName>.jpg images.
        threshold: cosine similarity threshold (lower = stricter).
        '''
        self.known_faces_dir = Path(known_faces_dir)
        self.threshold = threshold
        self.use_arcface = bool(use_arcface)
        self.app = None
        if self.use_arcface:
            try:
                from insightface.app import FaceAnalysis  # type: ignore
                self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=(160,160))
            except Exception:
                # Fallback: disable ArcFace if not installed or fails to initialize
                self.use_arcface = False
                self.app = None
        self.known_embeddings = []
        self.known_meta = []
        self._build_known_embeddings()

    def _embed(self, pil_img):
        if not self.use_arcface or self.app is None:
            # Fallback: Use simple feature extraction when ArcFace is not available
            try:
                from facenet_pytorch import MTCNN, InceptionResnetV1
                # Initialize models if not already done
                if not hasattr(self, 'mtcnn'):
                    self.mtcnn = MTCNN(keep_all=False, device='cpu')  # keep_all=False for single face
                if not hasattr(self, 'resnet'):
                    self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
                
                # Extract face and get embedding
                face_tensor = self.mtcnn(pil_img)
                if face_tensor is not None:
                    # Ensure correct tensor shape [3, 160, 160]
                    if face_tensor.dim() == 4:  # [1, 3, 160, 160]
                        face_tensor = face_tensor.squeeze(0)  # Remove batch dimension
                    
                    with torch.no_grad():
                        embedding = self.resnet(face_tensor.unsqueeze(0))
                        return embedding.squeeze().numpy() / np.linalg.norm(embedding.squeeze().numpy())
                return None
            except ImportError:
                # If facenet_pytorch is not available, use basic image features
                img_array = np.array(pil_img.convert('RGB').resize((160, 160)))
                # Simple feature extraction - convert to grayscale and flatten
                gray = np.mean(img_array, axis=2)
                features = gray.flatten() / 255.0
                return features / np.linalg.norm(features)
        
        img_np = np.array(pil_img.convert('RGB'))
        faces = self.app.get(img_np)
        if len(faces) == 0:
            return None
        # use first face embedding
        return faces[0].embedding / np.linalg.norm(faces[0].embedding)

    def _build_known_embeddings(self):
        if not self.known_faces_dir.exists():
            self.known_faces_dir.mkdir(parents=True, exist_ok=True)
            return
        for p in tqdm(list(self.known_faces_dir.iterdir()), desc="Loading known faces"):
            if p.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            name = p.stem
            if '_' in name:
                sid, fullname = name.split('_', 1)
            else:
                sid, fullname = name, name
            img = Image.open(p).convert('RGB')
            emb = self._embed(img)
            if emb is None:
                continue
            self.known_embeddings.append(emb)
            self.known_meta.append((sid, fullname))

    def match_faces(self, face_pil_list):
        results = []
        if len(self.known_embeddings) == 0:
            for _ in face_pil_list:
                results.append(('unknown', None, 0.0))
            return results
        
        # Group embeddings by student ID to avoid duplicates
        student_embeddings = {}
        student_names = {}
        for i, (sid, name) in enumerate(self.known_meta):
            if sid not in student_embeddings:
                student_embeddings[sid] = []
                student_names[sid] = name
            student_embeddings[sid].append(self.known_embeddings[i])
        
        # For each detected face, find the best match among all students
        for face in face_pil_list:
            emb = self._embed(face)
            if emb is None:
                results.append(('unknown', None, 0.0))
                continue
            
            best_student = None
            best_score = 0.0
            
            # Check each student's best photo
            for sid, embeddings in student_embeddings.items():
                # Calculate similarity with all photos of this student
                similarities = []
                for student_emb in embeddings:
                    sim = cosine_similarity([emb], [student_emb])[0][0]
                    similarities.append(sim)
                
                # Use the best similarity score for this student
                max_sim = max(similarities)
                if max_sim > best_score:
                    best_score = max_sim
                    best_student = (sid, student_names[sid])
            
            # Apply threshold and add result
            if best_score >= self.threshold and best_student:
                results.append((best_student[0], best_student[1], best_score))
            else:
                results.append(('unknown', None, best_score))
        
        # Post-process to remove duplicates and improve accuracy
        return self._remove_duplicates(results)
    
    def _remove_duplicates(self, results):
        """Remove duplicate recognitions of the same student"""
        seen_students = set()
        filtered_results = []
        
        # Sort by confidence score (highest first)
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        
        for student_id, name, score in sorted_results:
            if student_id == 'unknown':
                # Always keep unknown faces
                filtered_results.append((student_id, name, score))
            elif student_id not in seen_students:
                # Only keep the first (highest confidence) recognition of each student
                seen_students.add(student_id)
                filtered_results.append((student_id, name, score))
            # Skip duplicate recognitions of known students
        
        return filtered_results
