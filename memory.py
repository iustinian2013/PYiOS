# memory.py
import unicorn as uc
import logging

class SystemMemory:
    def __init__(self, uc_engine):
        self.uc = uc_engine
        self.mapped_regions = set()

    def map_region(self, address, size, protection=uc.UC_PROT_ALL):
        if size <= 0:
            return
            
        aligned_addr = address & ~0xFFF
        aligned_end = (address + size + 0xFFF) & ~0xFFF
        aligned_size = max(0x1000, aligned_end - aligned_addr)

        if aligned_addr in self.mapped_regions:
            return

        try:
            self.uc.mem_map(aligned_addr, aligned_size, protection)
            self.mapped_regions.add(aligned_addr)
            logging.info(f"[Memory Unit] Mapped region: 0x{aligned_addr:08X} (Size: 0x{aligned_size:08X}, Prot: {protection})")
        except uc.UcError as e:
            if "A memory region already exists" in str(e):
                self.mapped_regions.add(aligned_addr)
            else:
                logging.error(f"[Memory Unit Error] Failed to map 0x{aligned_addr:08X}: {e}")

    def write_payload(self, address, data):
        if not data:
            return
        try:
            self.uc.mem_write(address, data)
        except uc.UcError as e:
            logging.error(f"[Memory Unit Error] Failed payload write at 0x{address:08X}: {e}")