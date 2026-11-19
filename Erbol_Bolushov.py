class QueueArray:
    def __init__(self, size):
        self.arr = [None] * size
        self.topOfQueue = -1        # Acts as the rear/tail pointer
        self.beginningOfQueue = -1  # Acts as the front/head pointer
        print(f"The queue has been created with size of {size}")

    def isEmpty(self):
        """Checks whether the queue is empty."""
        return self.beginningOfQueue == -1

    def isFull(self):
        """Checks whether the queue is full (Circular Queue logic)."""
        if (self.topOfQueue + 1) % len(self.arr) == self.beginningOfQueue:
            return True
        elif self.beginningOfQueue == 0 and self.topOfQueue == len(self.arr) - 1:
            return True
        return False

    def enqueue(self, value):
        """Adds a new element to the rear of the queue."""
        if self.isFull():
            print("The Queue is full!")
        else:
            if self.isEmpty():
                self.beginningOfQueue = 0
            
            self.topOfQueue = (self.topOfQueue + 1) % len(self.arr)
            self.arr[self.topOfQueue] = value
            print(f"Successfully inserted {value} in the queue.")

    def dequeue(self):
        """Removes and returns an element from the front of the queue."""
        if self.isEmpty():
            print("The Queue is empty!")
            return None
        else:
            result = self.arr[self.beginningOfQueue]
            self.arr[self.beginningOfQueue] = None
            
            if self.beginningOfQueue == self.topOfQueue:
                # If there was only one element, reset the queue
                self.beginningOfQueue = -1
                self.topOfQueue = -1
            else:
                self.beginningOfQueue = (self.beginningOfQueue + 1) % len(self.arr)
                
            return result

    def peek(self):
        """Returns the front element without removing it."""
        if self.isEmpty():
            print("The Queue is empty!")
            return None
        else:
            return self.arr[self.beginningOfQueue]

    def deleteQueue(self):
        """Deletes the queue and releases the allocated memory."""
        self.arr = None
        self.topOfQueue = -1
        self.beginningOfQueue = -1
        print("The queue has been successfully deleted!")

    def emptyQueue(self):
        """Removes all elements from the queue without deleting the queue itself."""
        size = len(self.arr) if self.arr is not None else 0
        self.arr = [None] * size
        self.topOfQueue = -1
        self.beginningOfQueue = -1
        print("The queue has been successfully emptied!")


# --- Testing the Implementation ---
if __name__ == "__main__":
    qa = QueueArray(4)
    
    print(f"Is empty? {qa.isEmpty()}") # True
    
    qa.enqueue(10)
    qa.enqueue(20)
    qa.enqueue(30)
    qa.enqueue(40)
    
    print(f"Is full? {qa.isFull()}")   # True
    print(f"Peek front: {qa.peek()}")  # 10
    
    print(f"Dequeued: {qa.dequeue()}") # 10
    print(f"Peek new front: {qa.peek()}") # 20
    
    qa.enqueue(50) # Tests circular wrap-around
    print(f"Is full again? {qa.isFull()}") # True