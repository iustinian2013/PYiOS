# bridge.py
import logging

class IOSRuntimeBridge:
    def __init__(self, uc_engine, gpu_instance):
        self.uc = uc_engine
        self.gpu = gpu_instance
        self.classes = {}
        self.selectors = {}
        logging.info("[Runtime Bridge] Initialized Objective-C Runtime and Framework Bridge layer.")

    def register_hooks(self, cpu_core):
        logging.info("[Runtime Bridge] Hook registration routed safely through ARMCpuCore.")

    def register_class(self, class_name, super_class=None):
        class_struct = {
            'name': class_name,
            'super': super_class,
            'methods': {}
        }
        self.classes[class_name] = class_struct
        logging.debug(f"[ObjC Runtime] Registered class: {class_name}")
        return class_struct

    def register_selector(self, sel_name):
        if sel_name not in self.selectors:
            sel_id = len(self.selectors) + 1
            self.selectors[sel_name] = sel_id
        return self.selectors[sel_name]

    def resolve_msg_send(self, receiver, selector_ptr):
        try:
            sel_bytes = None
            for length in (256, 128, 64, 32):   # long selectors are common (buttonWithFrame:normalImage:...)
                try:
                    sel_bytes = bytes(self.uc.mem_read(selector_ptr, length))
                    break
                except Exception:
                    continue
            if sel_bytes is None:
                raise ValueError("selector unreadable")
            sel_name = sel_bytes.split(b'\x00')[0].decode('utf-8', errors='ignore')
            logging.debug(f"[ObjC msgSend] Receiver: 0x{receiver:08X} | Selector: {sel_name}")
            return sel_name
        except Exception:
            logging.warning(f"[ObjC msgSend] Failed to resolve selector pointer at 0x{selector_ptr:08X}")
            return None