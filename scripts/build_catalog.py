import argparse
import sys
import signal
from pathlib import Path
from typing import List, Dict, Optional
import logging
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.product_counting_system import ProductCountingSystem
from src.utils.logging_utils import setup_logging, PerformanceLogger
from src.utils.file_utils import ensure_dir, save_json, list_images
from src.exceptions.core_exceptions import SystemInitializationError, CatalogError
from config.settings import get_settings


class CatalogBuilder:
    """Handles the complete catalog building process"""
    
    def __init__(self, config_env: str = 'development'):
        self.settings = get_settings(config_env)
        self.logger = logging.getLogger(__name__)
        self.system = None
        self.stats = {
            'products_found': 0,
            'products_added': 0,
            'products_failed': 0,
            'total_images': 0,
            'total_embeddings': 0,
            'processing_time': 0
        }
    
    def initialize_system(self):
        """Initialize the product counting system"""
        try:
            self.logger.info("Initializing product counting system...")
            with PerformanceLogger(self.logger, "System initialization"):
                self.system = ProductCountingSystem(config=self.settings.dict())
            
            self.logger.info("System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise SystemInitializationError(f"Failed to initialize system: {e}")
    
    def scan_images_directory(self, images_dir: Path) -> Dict[str, List[Path]]:
        """Scan directory and organize products by folder"""
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        products = {}
        
        self.logger.info(f"Scanning directory: {images_dir}")
        
        for item in images_dir.iterdir():
            if not item.is_dir():
                continue
            
            product_id = item.name
            image_files = list_images(item)
            
            if image_files:
                products[product_id] = image_files
                self.stats['products_found'] += 1
                self.stats['total_images'] += len(image_files)
                
                self.logger.info(f"Found product '{product_id}': {len(image_files)} images")
            else:
                self.logger.warning(f"Skipping '{product_id}': no valid images found")
        
        self.logger.info(f"Scan complete: {len(products)} products, {self.stats['total_images']} total images")
        return products
    
    def build_catalog_from_products(self, products: Dict[str, List[Path]], 
                                  force_rebuild: bool = False) -> bool:
        """Build catalog from discovered products"""
        
        if not products:
            self.logger.warning("No products found to add to catalog")
            return False
        
        start_time = time.time()
        
        try:
            with PerformanceLogger(self.logger, f"Building catalog for {len(products)} products"):
                
                for product_id, image_paths in products.items():
                    try:
                        # Check if product already exists
                        if not force_rebuild and self.system.catalog_manager.get_product(product_id):
                            self.logger.info(f"Skipping existing product: {product_id}")
                            continue
                        
                        # Prepare product name
                        product_name = product_id.replace('_', ' ').replace('-', ' ').title()
                        
                        # Add product to catalog
                        self.logger.info(f"Processing product: {product_id}")
                        
                        product_info = self.system.add_product_to_catalog(
                            product_id=product_id,
                            name=product_name,
                            image_paths=[str(img) for img in image_paths],
                            category='retail_product',  # Default category
                            description=f"Product {product_name} with {len(image_paths)} reference images"
                        )
                        
                        self.stats['products_added'] += 1
                        self.stats['total_embeddings'] += len(product_info.embedding_indices)
                        
                        self.logger.info(f"Added product: {product_id} ({len(product_info.embedding_indices)} embeddings)")
                        
                    except Exception as e:
                        self.stats['products_failed'] += 1
                        self.logger.error(f"Failed to add product {product_id}: {e}")
                        continue
                
                self.stats['processing_time'] = time.time() - start_time
                
                if self.stats['products_added'] > 0:
                    self.logger.info("Catalog building completed successfully")
                    return True
                else:
                    self.logger.warning("No products were added to catalog")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Catalog building failed: {e}")
            raise CatalogError(f"Failed to build catalog: {e}")
    
    def save_catalog_and_system(self, output_dir: Path) -> None:
        """Save the complete system state"""
        
        ensure_dir(output_dir)
        
        try:
            with PerformanceLogger(self.logger, "Saving system state"):
                # Save system state (includes catalog, embeddings, indices)
                self.system.save_system_state(output_dir)
                
                # Save build statistics
                stats_file = output_dir / "build_stats.json"
                build_info = {
                    'build_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'build_stats': self.stats,
                    'system_config': self.settings.dict(),
                    'system_statistics': self.system.get_system_statistics()
                }
                
                save_json(build_info, stats_file)
                
                self.logger.info(f"System state saved to: {output_dir}")
                self.logger.info(f"Build statistics saved to: {stats_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to save system state: {e}")
            raise
    
    def print_summary(self) -> None:
        """Print build summary"""
        
        print("\n" + "="*60)
        print("CATALOG BUILD SUMMARY")
        print("="*60)
        
        print(f"Products found:      {self.stats['products_found']}")
        print(f"Products added:      {self.stats['products_added']}")
        print(f"Products failed:     {self.stats['products_failed']}")
        print(f"Total images:        {self.stats['total_images']}")
        print(f"Total embeddings:    {self.stats['total_embeddings']}")
        print(f"Processing time:     {self.stats['processing_time']:.1f}s")
        
        if self.stats['products_added'] > 0:
            print(f"Avg time per product: {self.stats['processing_time']/self.stats['products_added']:.1f}s")
            print(f"Success rate:        {self.stats['products_added']/self.stats['products_found']:.1%}")
        
        print("="*60)
        
        if self.stats['products_added'] == 0:
            print("No products were successfully added to catalog")
        elif self.stats['products_failed'] == 0:
            print("All products processed successfully!")
        else:
            print(f"{self.stats['products_failed']} products failed to process")


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    logging.getLogger(__name__).info("Build process interrupted by user")
    sys.exit(1)

def main():
    """Main function"""
    
    # Define default configuration
    images_dir = Path("data/reference_images")
    output_dir = Path("results")
    config_env = "development"
    log_level = "INFO"
    force_rebuild = False
    dry_run = False
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup logging
    log_file = output_dir / "build_catalog.log" if not dry_run else None
    setup_logging(
        log_level=log_level,
        log_file=log_file
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting catalog build process")
        logger.info(f"Images directory: {images_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Configuration: {config_env}")
        logger.info(f"Force rebuild: {force_rebuild}")
        logger.info(f"Dry run: {dry_run}")
        
        # Initialize builder
        builder = CatalogBuilder(config_env=config_env)
        
        # Scan images directory
        products = builder.scan_images_directory(images_dir)
        
        if not products:
            logger.error("No products found in images directory")
            return 1
        
        # Dry run mode
        if dry_run:
            logger.info("Dry run mode - scanning only")
            builder.print_summary()
            return 0
        
        # Initialize system
        if not builder.initialize_system():
            return 1
        
        # Build catalog
        success = builder.build_catalog_from_products(
            products=products,
            force_rebuild=force_rebuild
        )
        
        if success:
            # Save system state
            builder.save_catalog_and_system(output_dir)
            
            # Print summary
            builder.print_summary()
            
            logger.info("Catalog build completed successfully!")
            return 0
        else:
            logger.error("Catalog build failed")
            return 1
        
    except KeyboardInterrupt:
        logger.info("Build process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Build process failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())