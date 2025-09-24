import json
import numpy as np
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.exceptions.core_exceptions import CatalogError
from src.utils.validation import validate_product_data
from src.utils.file_utils import ensure_dir, load_json, save_json, list_images
from src.utils.image_utils import validate_image, load_image
from src.utils.logging_utils import PerformanceLogger


logger = logging.getLogger(__name__)

@dataclass
class ProductInfo:
    """Product information data class"""
    product_id: str
    name: str
    reference_images: List[str]
    embedding_indices: List[int]
    category: Optional[str] = None
    description: Optional[str] = None
    barcode: Optional[str] = None
    price: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.metadata is None:
            self.metadata = {}
    
    def update(self) -> None:
        """Update the timestamp"""
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProductInfo':
        """Create from dictionary"""
        return cls(**data)
    
class ProductCatalogManager:
    """Comprehensive product catalog management system"""
    
    def __init__(self, catalog_file: Optional[Path] = None):
        self.catalog_file = catalog_file
        self.products: Dict[str, ProductInfo] = {}
        self.categories: Dict[str, List[str]] = {}  # category -> [product_ids]
        self.embeddings_map: Dict[int, str] = {}    # embedding_index -> product_id
        
        logger.info("Initialized product catalog manager")
        
        if catalog_file and catalog_file.exists():
            self.load_catalog(catalog_file)
    
    def add_product(self, product_id: str, name: str, reference_images: List[str],
                   category: Optional[str] = None, **kwargs) -> ProductInfo:
        """Add new product to catalog"""
        
        # Validate inputs
        product_data = {
            'product_id': product_id,
            'name': name,
            'reference_images': reference_images,
            **kwargs
        }
        
        validation_errors = validate_product_data(product_data)
        if validation_errors:
            raise CatalogError(f"Product validation failed: {validation_errors}")
        
        # Check if product already exists
        if product_id in self.products:
            raise CatalogError(f"Product already exists: {product_id}")
        
        # Validate reference images
        valid_images = []
        for img_path in reference_images:
            if validate_image(img_path):
                valid_images.append(str(Path(img_path).resolve()))
            else:
                logger.warning(f"Invalid reference image: {img_path}")
        
        if not valid_images:
            raise CatalogError(f"No valid reference images for product: {product_id}")
        
        # Create product info
        product_info = ProductInfo(
            product_id=product_id,
            name=name,
            reference_images=valid_images,
            embedding_indices=[],  # Will be set when embeddings are added
            category=category,
            **kwargs
        )
        
        # Add to catalog
        self.products[product_id] = product_info
        
        # Add to category mapping
        if category:
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(product_id)
        
        logger.info(f"Added product to catalog: {product_id}")
        return product_info
    
    def update_product(self, product_id: str, **kwargs) -> ProductInfo:
        """Update existing product"""
        if product_id not in self.products:
            raise CatalogError(f"Product not found: {product_id}")
        
        product_info = self.products[product_id]
        
        # Update allowed fields
        updateable_fields = ['name', 'reference_images', 'category', 'description', 
                           'barcode', 'price', 'metadata']
        
        for field, value in kwargs.items():
            if field in updateable_fields:
                setattr(product_info, field, value)
        
        product_info.update()  # Update timestamp
        
        logger.info(f"Updated product: {product_id}")
        return product_info
    
    def remove_product(self, product_id: str) -> None:
        """Remove product from catalog"""
        if product_id not in self.products:
            raise CatalogError(f"Product not found: {product_id}")
        
        product_info = self.products[product_id]
        
        # Remove from category mapping
        if product_info.category and product_info.category in self.categories:
            if product_id in self.categories[product_info.category]:
                self.categories[product_info.category].remove(product_id)
                if not self.categories[product_info.category]:
                    del self.categories[product_info.category]
        
        # Remove from embeddings mapping
        for idx in product_info.embedding_indices:
            if idx in self.embeddings_map:
                del self.embeddings_map[idx]
        
        # Remove product
        del self.products[product_id]
        
        logger.info(f"Removed product: {product_id}")
    
    def get_product(self, product_id: str) -> Optional[ProductInfo]:
        """Get product information"""
        return self.products.get(product_id)
    
    def list_products(self, category: Optional[str] = None) -> List[ProductInfo]:
        """List products, optionally filtered by category"""
        if category:
            product_ids = self.categories.get(category, [])
            return [self.products[pid] for pid in product_ids]
        else:
            return list(self.products.values())
    
    def search_products(self, query: str) -> List[ProductInfo]:
        """Search products by name or ID"""
        query_lower = query.lower()
        matches = []
        
        for product in self.products.values():
            if (query_lower in product.product_id.lower() or 
                query_lower in product.name.lower()):
                matches.append(product)
        
        return matches
    
    def add_embedding_index(self, product_id: str, embedding_index: int) -> None:
        """Associate embedding index with product"""
        if product_id not in self.products:
            raise CatalogError(f"Product not found: {product_id}")
        
        self.products[product_id].embedding_indices.append(embedding_index)
        self.embeddings_map[embedding_index] = product_id
        self.products[product_id].update()
    
    def get_product_by_embedding_index(self, embedding_index: int) -> Optional[ProductInfo]:
        """Get product by embedding index"""
        product_id = self.embeddings_map.get(embedding_index)
        return self.products.get(product_id) if product_id else None
    
    def build_from_directory(self, images_dir: Path, pattern: str = "*") -> int:
        """Build catalog from directory structure"""
        images_dir = Path(images_dir)
        if not images_dir.exists():
            raise CatalogError(f"Directory not found: {images_dir}")
        
        products_added = 0
        
        with PerformanceLogger(logger, f"Building catalog from {images_dir}"):
            # Look for subdirectories (each is a product)
            for product_dir in images_dir.iterdir():
                if not product_dir.is_dir():
                    continue
                
                product_id = product_dir.name
                
                # Skip if product already exists
                if product_id in self.products:
                    logger.info(f"Skipping existing product: {product_id}")
                    continue
                
                # Find reference images
                reference_images = list_images(product_dir)
                
                if not reference_images:
                    logger.warning(f"No images found for product: {product_id}")
                    continue
                
                try:
                    # Create product with directory name as both ID and name
                    self.add_product(
                        product_id=product_id,
                        name=product_id.replace('_', ' ').title(),
                        reference_images=[str(img) for img in reference_images]
                    )
                    products_added += 1
                    
                except Exception as e:
                    logger.error(f"Failed to add product {product_id}: {e}")
        
        logger.info(f"Built catalog: {products_added} products added")
        return products_added
    
    def validate_catalog(self) -> List[str]:
        """Validate catalog integrity"""
        issues = []
        
        # Check products
        for product_id, product_info in self.products.items():
            # Validate reference images exist
            for img_path in product_info.reference_images:
                if not Path(img_path).exists():
                    issues.append(f"Missing reference image: {img_path} for product {product_id}")
            
            # Validate embedding indices
            for idx in product_info.embedding_indices:
                if idx not in self.embeddings_map:
                    issues.append(f"Orphaned embedding index {idx} for product {product_id}")
                elif self.embeddings_map[idx] != product_id:
                    issues.append(f"Embedding index {idx} mapping mismatch for product {product_id}")
        
        # Check categories
        for category, product_ids in self.categories.items():
            for product_id in product_ids:
                if product_id not in self.products:
                    issues.append(f"Category {category} references missing product: {product_id}")
                elif self.products[product_id].category != category:
                    issues.append(f"Category mismatch for product {product_id}")
        
        if issues:
            logger.warning(f"Found {len(issues)} catalog validation issues")
        else:
            logger.info("Catalog validation passed")
        
        return issues
    
    def get_statistics(self) -> Dict[str, any]:
        """Get catalog statistics"""
        stats = {
            'total_products': len(self.products),
            'total_categories': len(self.categories),
            'total_reference_images': sum(len(p.reference_images) for p in self.products.values()),
            'total_embeddings': len(self.embeddings_map),
            'products_by_category': {cat: len(pids) for cat, pids in self.categories.items()},
            'avg_images_per_product': 0,
            'products_without_embeddings': 0
        }
        
        if self.products:
            stats['avg_images_per_product'] = stats['total_reference_images'] / stats['total_products']
            stats['products_without_embeddings'] = sum(
                1 for p in self.products.values() if not p.embedding_indices
            )
        
        return stats
    
    def save_catalog(self, filepath: Optional[Path] = None) -> None:
        """Save catalog to JSON file"""
        if filepath is None:
            if self.catalog_file is None:
                raise CatalogError("No catalog file specified")
            filepath = self.catalog_file
        
        try:
            ensure_dir(Path(filepath).parent)
            
            catalog_data = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'products': {pid: pinfo.to_dict() for pid, pinfo in self.products.items()},
                'categories': self.categories,
                'embeddings_map': {str(k): v for k, v in self.embeddings_map.items()},
                'statistics': self.get_statistics()
            }
            
            save_json(catalog_data, filepath, indent=2)
            logger.info(f"Saved catalog to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save catalog: {e}")
            raise CatalogError(f"Save failed: {e}")
    
    def load_catalog(self, filepath: Path) -> None:
        """Load catalog from JSON file"""
        try:
            catalog_data = load_json(filepath)
            
            # Load products
            self.products.clear()
            for product_id, product_data in catalog_data.get('products', {}).items():
                self.products[product_id] = ProductInfo.from_dict(product_data)
            
            # Load categories
            self.categories = catalog_data.get('categories', {})
            
            # Load embeddings mapping (convert string keys back to int)
            embeddings_map = catalog_data.get('embeddings_map', {})
            self.embeddings_map = {int(k): v for k, v in embeddings_map.items()}
            
            self.catalog_file = filepath
            
            logger.info(f"Loaded catalog from {filepath}: {len(self.products)} products")
            
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            raise CatalogError(f"Load failed: {e}")
    
    def export_catalog(self, filepath: Path, format: str = "json") -> None:
        """Export catalog in different formats"""
        try:
            ensure_dir(Path(filepath).parent)
            
            if format.lower() == "json":
                self.save_catalog(filepath)
                
            elif format.lower() == "csv":
                import csv
                
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow([
                        'product_id', 'name', 'category', 'reference_images_count',
                        'embeddings_count', 'created_at', 'updated_at'
                    ])
                    
                    # Data
                    for product in self.products.values():
                        writer.writerow([
                            product.product_id,
                            product.name,
                            product.category or '',
                            len(product.reference_images),
                            len(product.embedding_indices),
                            product.created_at,
                            product.updated_at
                        ])
                
                logger.info(f"Exported catalog to CSV: {filepath}")
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export catalog: {e}")
            raise CatalogError(f"Export failed: {e}")