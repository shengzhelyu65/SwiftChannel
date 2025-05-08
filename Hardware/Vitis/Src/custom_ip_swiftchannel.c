#include "xaxidma.h"
#include "xparameters.h"
#include "xil_exception.h"
#include "xdebug.h"
#include "xil_util.h"
#include "xscugic.h"
#include <stdio.h>
#include "xtime_l.h" // Include XTime library for timestamping

#include "xswiftchannel.h"
#include "rx_grid.h"
#include "h_output.h"

/************************** Constant Definitions *****************************/

/*
 * Device hardware build related constants.
 */
#define DMA_DEV_ID		XPAR_AXIDMA_0_DEVICE_ID

#define DDR_BASE_ADDR	XPAR_PSU_DDR_0_S_AXI_BASEADDR

#define MEM_BASE_ADDR		(DDR_BASE_ADDR + 0x1000000)

#define RX_INTR_ID		XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID
#define TX_INTR_ID		XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID

#define TX_BUFFER_BASE		(MEM_BASE_ADDR + 0x00100000)
#define RX_BUFFER_BASE		(MEM_BASE_ADDR + 0x00300000)
#define RX_BUFFER_HIGH		(MEM_BASE_ADDR + 0x004FFFFF)

#define INTC_DEVICE_ID          XPAR_SCUGIC_SINGLE_DEVICE_ID

#define INTC		XScuGic
#define INTC_HANDLER	XScuGic_InterruptHandler


#define CUSTOM_IP_DEVICE_ID 		XPAR_SWIFTCHANNEL_0_DEVICE_ID

/* Timeout loop counter for reset
 */
#define RESET_TIMEOUT_COUNTER	10000

#define TEST_START_VALUE	0x3
/*
 * Buffer and Buffer Descriptor related constant definition
 */

#define POLL_TIMEOUT_COUNTER    1000000U
#define NUMBER_OF_EVENTS	1

/*
 * Test-Specific definition
 */
#define INPUT_HEIGHT  108
#define INPUT_WIDTH   16
#define INPUT_CHAN    2

#define OUTPUT_HEIGHT 432
#define OUTPUT_WIDTH  128
#define OUTPUT_CHAN   2

#define SIZE_OF_TDADA 4

#define INPUT_SIZE_LEN      (INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHAN)
#define INPUT_SIZE_BYTES    (INPUT_SIZE_LEN * SIZE_OF_TDADA)
#define OUTPUT_SIZE_LEN     (OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHAN)
#define OUTPUT_SIZE_BYTES   (OUTPUT_SIZE_LEN * SIZE_OF_TDADA)

/************************** Function Prototypes ******************************/
#ifndef DEBUG
extern void xil_printf(const char *format, ...);
#endif

static int CheckData(int Length);
static void TxIntrHandler(void *Callback);
static void RxIntrHandler(void *Callback);

static int SetupIntrSystem(INTC *IntcInstancePtr,
			   XAxiDma *AxiDmaPtr, u16 TxIntrId, u16 RxIntrId);
static void DisableIntrSystem(INTC *IntcInstancePtr,
			      u16 TxIntrId, u16 RxIntrId);

/************************** Variable Definitions *****************************/
/*
 * Device instance definitions
 */
static XAxiDma AxiDma;		/* Instance of the XAxiDma */
static XSwiftchannel IP;         /* Instance of the IP */
static INTC Intc;	/* Instance of the Interrupt Controller */
/*
 * Flags interrupt handlers use to notify the application context the events.
 */
volatile u32 TxDone;
volatile u32 RxDone;
volatile u32 Error;

int initCustomIP(XSwiftchannel* instance, XSwiftchannel_Config* IPConfig)
{
    int Status;

    Status = XSwiftchannel_CfgInitialize(instance, IPConfig);
	if (Status != XST_SUCCESS) {
		xil_printf("No IP config found for %d\r\n", Status);
		return XST_FAILURE;
	}
    if (XSwiftchannel_IsReady(instance)) {
        xil_printf("IP core is ready!\r\n");
    } else {
        xil_printf("IP core is not ready!\r\n");
    }
    if (XSwiftchannel_IsIdle(instance)) {
        xil_printf("IP core is idle!\r\n");
    } else {
        xil_printf("IP core is not idle!\r\n");
    }

	XSwiftchannel_InterruptGlobalDisable(instance);
	XSwiftchannel_EnableAutoRestart(instance);

    XSwiftchannel_DisableAutoRestart(instance);
	XSwiftchannel_Start(instance);

	return XST_SUCCESS;
}

/*****************************************************************************/
/**
*
* Main function
*
* This function is the main entry of the interrupt test. It does the following:
*	Set up the output terminal if UART16550 is in the hardware build
*	Initialize the DMA engine
*	Set up Tx and Rx channels
*	Set up the interrupt system for the Tx and Rx interrupts
*	Submit a transfer
*	Wait for the transfer to finish
*	Check transfer status
*	Disable Tx and Rx interrupts
*	Print test status and exit
*
* @param	None
*
* @return
*		- XST_SUCCESS if example finishes successfully
*		- XST_FAILURE if example fails.
*
* @note		None.
*
******************************************************************************/
int main(void)
{
	XTime StartTime, EndTime; // Variables to store timestamps
	u64 Latency;             // Variable to store calculated latency

	int Status;
	int Counter;

	XAxiDma_Config *Config;
	XSwiftchannel_Config *IPConfig;

	float *TxBufferPtr;
	float *RxBufferPtr;

	TxBufferPtr = (float *)TX_BUFFER_BASE ;
	RxBufferPtr = (float *)RX_BUFFER_BASE;

	xil_printf("\r\n--- Entering main() --- \r\n");

	Config = XAxiDma_LookupConfig(DMA_DEV_ID);
	if (!Config) {
		xil_printf("No config found for %d\r\n", DMA_DEV_ID);

		return XST_FAILURE;
	}

	IPConfig = XSwiftchannel_LookupConfig(CUSTOM_IP_DEVICE_ID);
	if (!IPConfig) {
		xil_printf("No config found for %d\r\n", CUSTOM_IP_DEVICE_ID);
		goto Done;
	}

	/* Initialize DMA engine */
	Status = XAxiDma_CfgInitialize(&AxiDma, Config);
	if (Status != XST_SUCCESS) {
		xil_printf("Initialization failed %d\r\n", Status);
		return XST_FAILURE;
	}
	if (XAxiDma_HasSg(&AxiDma)) {
		xil_printf("Device configured as SG mode \r\n");
		return XST_FAILURE;
	}

	/* Set up Interrupt system  */
	Status = SetupIntrSystem(&Intc, &AxiDma, TX_INTR_ID, RX_INTR_ID);
	if (Status != XST_SUCCESS) {
		xil_printf("Failed intr setup\r\n");
		return XST_FAILURE;
	}

	/* Disable all interrupts before setup */
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
			    XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
			    XAXIDMA_DEVICE_TO_DMA);

	/* Enable all interrupts */
	XAxiDma_IntrEnable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
			   XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_IntrEnable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
			   XAXIDMA_DEVICE_TO_DMA);

	/* Initialize flags before start transfer test  */
	TxDone = 0;
	RxDone = 0;
	Error = 0;

	/* Flush the buffers before the DMA transfer, in case the Data Cache
	 * is enabled
	 */
	Xil_DCacheFlushRange((UINTPTR)TxBufferPtr, INPUT_SIZE_BYTES);
	Xil_DCacheFlushRange((UINTPTR)RxBufferPtr, OUTPUT_SIZE_BYTES);

	/* Disable all cache */
	Xil_DCacheDisable();
	Xil_ICacheDisable();

	Counter = 0;
	for (int h = 0; h < INPUT_HEIGHT; h++) {
		for (int w = 0; w < INPUT_WIDTH; w++) {
			for (int c = 0; c < INPUT_CHAN; c++) {
				TxBufferPtr[Counter] = TEST_DATA[h][w][c];
				Counter++;
			}
		}
	}

	xil_printf("Successfully write %d packets to TxBuffer right now\r\n", Counter);

	Status = initCustomIP(&IP, IPConfig);
	if (Status != XST_SUCCESS)
	{
		xil_printf("IP initialization error %d\r\n", Status);
		goto Done;
	}

	/* Record the start time */
	XTime_GetTime(&StartTime);

	/* Send a packet */
	Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR) RxBufferPtr, OUTPUT_SIZE_BYTES, XAXIDMA_DEVICE_TO_DMA);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR) TxBufferPtr, INPUT_SIZE_BYTES, XAXIDMA_DMA_TO_DEVICE);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	Status = Xil_WaitForEventSet(POLL_TIMEOUT_COUNTER, NUMBER_OF_EVENTS, &Error);
	if (Status == XST_SUCCESS) {
		if (!TxDone) {
			xil_printf("Transmit error %d\r\n", Status);
			goto Done;
		} else if (Status == XST_SUCCESS && !RxDone) {
			xil_printf("Receive error %d\r\n", Status);
			goto Done;
		}
	}

	/*
	 * Wait for TX done or timeout
	 */
	Status = Xil_WaitForEventSet(POLL_TIMEOUT_COUNTER, NUMBER_OF_EVENTS, &TxDone);
	if (Status != XST_SUCCESS) {
		xil_printf("Transmit failed %d\r\n", Status);
		goto Done;
	}
	xil_printf("Successfully transmit the data to IP\r\n");

	/*
	 * Wait for RX done or timeout
	 */
	Status = Xil_WaitForEventSet(POLL_TIMEOUT_COUNTER, NUMBER_OF_EVENTS, &RxDone);
	if (Status != XST_SUCCESS) {
		xil_printf("Receive failed %d\r\n", Status);
		goto Done;
	}

	/* Record the end time */
	XTime_GetTime(&EndTime);

	/* Calculate latency in microseconds */
	Latency = ((EndTime - StartTime) * 1000000) / COUNTS_PER_SECOND;
	xil_printf("Total time consuming of IP: %llu microseconds\r\n", Latency);

	xil_printf("Successfully receive the data from IP\r\n");
	xil_printf("Checking data now\r\n");

	xil_printf("Checking data now\r\n");

	/*
	 * Test finished, check data
	 */
	Status = CheckData(OUTPUT_SIZE_LEN);
	if (Status != XST_SUCCESS) {
		xil_printf("Data check failed\r\n");
		goto Done;
	}

	xil_printf("Successfully ran AXI DMA interrupt Example\r\n");

	/* Disable TX and RX Ring interrupts and return success */
	DisableIntrSystem(&Intc, TX_INTR_ID, RX_INTR_ID);

Done:
	xil_printf("--- Exiting main() --- \r\n");

	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	return XST_SUCCESS;
}

/*****************************************************************************/
/*
*
* This function checks data buffer after the DMA transfer is finished.
*
* We use the static tx/rx buffers.
*
* @param	Length is the length to check
* @param	StartValue is the starting value of the first byte
*
* @return
*		- XST_SUCCESS if validation is successful
*		- XST_FAILURE if validation is failure.
*
* @note		None.
*
******************************************************************************/
static int CheckData(int Length)
{
	float *RxPacket;
	int Index = 0;
    int GT_row = 0;
    int GT_chan = 0;
    int GT_col = 0;
    int Offset = 0;
    float Compare;

	RxPacket = (float *) RX_BUFFER_BASE;

	/* Invalidate the DestBuffer before receiving the data, in case the
	 * Data Cache is enabled
	 */
	// Xil_DCacheInvalidateRange((UINTPTR)RxPacket, Length * sizeof(float));

	for (Index = 0; Index < Length; Index++) {
        GT_row = Index / (OUTPUT_WIDTH * OUTPUT_CHAN);
        Offset = Index % (OUTPUT_WIDTH * OUTPUT_CHAN);
        GT_col = Offset / OUTPUT_CHAN;
        GT_chan = Offset % OUTPUT_CHAN;

        if (Index < 10)
        {
        	printf("Data display %d: %f/%f\r\n", Index, RxPacket[Index], DATA_GT[GT_row][GT_col][GT_chan]);
        }

        Compare = RxPacket[Index] - DATA_GT[GT_row][GT_col][GT_chan];

		if (Compare < -0.01f || Compare > 0.01f)
        {
			printf("Data error %d: %f/%f\r\n", Index, RxPacket[Index], DATA_GT[GT_row][GT_col][GT_chan]);
			return XST_FAILURE;
		}
	}

	return XST_SUCCESS;
}

/*****************************************************************************/
/*
*
* This is the DMA TX Interrupt handler function.
*
* It gets the interrupt status from the hardware, acknowledges it, and if any
* error happens, it resets the hardware. Otherwise, if a completion interrupt
* is present, then sets the TxDone.flag
*
* @param	Callback is a pointer to TX channel of the DMA engine.
*
* @return	None.
*
* @note		None.
*
******************************************************************************/
static void TxIntrHandler(void *Callback)
{

	u32 IrqStatus;
	int TimeOut;
	XAxiDma *AxiDmaInst = (XAxiDma *)Callback;

	/* Read pending interrupts */
	IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DMA_TO_DEVICE);

	/* Acknowledge pending interrupts */


	XAxiDma_IntrAckIrq(AxiDmaInst, IrqStatus, XAXIDMA_DMA_TO_DEVICE);

	/*
	 * If no interrupt is asserted, we do not do anything
	 */
	if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK)) {

		return;
	}

	/*
	 * If error interrupt is asserted, raise error flag, reset the
	 * hardware to recover from the error, and return with no further
	 * processing.
	 */
	if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK)) {

		Error = 1;

		/*
		 * Reset should never fail for transmit channel
		 */
		XAxiDma_Reset(AxiDmaInst);

		TimeOut = RESET_TIMEOUT_COUNTER;

		while (TimeOut) {
			if (XAxiDma_ResetIsDone(AxiDmaInst)) {
				break;
			}

			TimeOut -= 1;
		}

		return;
	}

	/*
	 * If Completion interrupt is asserted, then set the TxDone flag
	 */
	if ((IrqStatus & XAXIDMA_IRQ_IOC_MASK)) {

		TxDone = 1;
	}
}

/*****************************************************************************/
/*
*
* This is the DMA RX interrupt handler function
*
* It gets the interrupt status from the hardware, acknowledges it, and if any
* error happens, it resets the hardware. Otherwise, if a completion interrupt
* is present, then it sets the RxDone flag.
*
* @param	Callback is a pointer to RX channel of the DMA engine.
*
* @return	None.
*
* @note		None.
*
******************************************************************************/
static void RxIntrHandler(void *Callback)
{
	u32 IrqStatus;
	int TimeOut;
	XAxiDma *AxiDmaInst = (XAxiDma *)Callback;

	/* Read pending interrupts */
	IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DEVICE_TO_DMA);

	/* Acknowledge pending interrupts */
	XAxiDma_IntrAckIrq(AxiDmaInst, IrqStatus, XAXIDMA_DEVICE_TO_DMA);

	/*
	 * If no interrupt is asserted, we do not do anything
	 */
	if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK)) {
		return;
	}

	/*
	 * If error interrupt is asserted, raise error flag, reset the
	 * hardware to recover from the error, and return with no further
	 * processing.
	 */
	if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK)) {

		Error = 1;

		/* Reset could fail and hang
		 * NEED a way to handle this or do not call it??
		 */
		XAxiDma_Reset(AxiDmaInst);

		TimeOut = RESET_TIMEOUT_COUNTER;

		while (TimeOut) {
			if (XAxiDma_ResetIsDone(AxiDmaInst)) {
				break;
			}

			TimeOut -= 1;
		}

		return;
	}

	/*
	 * If completion interrupt is asserted, then set RxDone flag
	 */
	if ((IrqStatus & XAXIDMA_IRQ_IOC_MASK)) {

		RxDone = 1;
	}
}

/*****************************************************************************/
/*
*
* This function setups the interrupt system so interrupts can occur for the
* DMA, it assumes INTC component exists in the hardware system.
*
* @param	IntcInstancePtr is a pointer to the instance of the INTC.
* @param	AxiDmaPtr is a pointer to the instance of the DMA engine
* @param	TxIntrId is the TX channel Interrupt ID.
* @param	RxIntrId is the RX channel Interrupt ID.
*
* @return
*		- XST_SUCCESS if successful,
*		- XST_FAILURE.if not successful
*
* @note		None.
*
******************************************************************************/
static int SetupIntrSystem(INTC *IntcInstancePtr,
			   XAxiDma *AxiDmaPtr, u16 TxIntrId, u16 RxIntrId)
{
	int Status;

	XScuGic_Config *IntcConfig;

	/*
	 * Initialize the interrupt controller driver so that it is ready to
	 * use.
	 */
	IntcConfig = XScuGic_LookupConfig(INTC_DEVICE_ID);
	if (NULL == IntcConfig) {
		return XST_FAILURE;
	}

	Status = XScuGic_CfgInitialize(IntcInstancePtr, IntcConfig,
				       IntcConfig->CpuBaseAddress);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}


	XScuGic_SetPriorityTriggerType(IntcInstancePtr, TxIntrId, 0xA0, 0x3);

	XScuGic_SetPriorityTriggerType(IntcInstancePtr, RxIntrId, 0xA0, 0x3);
	/*
	 * Connect the device driver handler that will be called when an
	 * interrupt for the device occurs, the handler defined above performs
	 * the specific interrupt processing for the device.
	 */
	Status = XScuGic_Connect(IntcInstancePtr, TxIntrId,
				 (Xil_InterruptHandler)TxIntrHandler,
				 AxiDmaPtr);
	if (Status != XST_SUCCESS) {
		return Status;
	}

	Status = XScuGic_Connect(IntcInstancePtr, RxIntrId,
				 (Xil_InterruptHandler)RxIntrHandler,
				 AxiDmaPtr);
	if (Status != XST_SUCCESS) {
		return Status;
	}

	XScuGic_Enable(IntcInstancePtr, TxIntrId);
	XScuGic_Enable(IntcInstancePtr, RxIntrId);

	/* Enable interrupts from the hardware */

	Xil_ExceptionInit();
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
				     (Xil_ExceptionHandler)INTC_HANDLER,
				     (void *)IntcInstancePtr);

	Xil_ExceptionEnable();

	return XST_SUCCESS;
}

/*****************************************************************************/
/**
*
* This function disables the interrupts for DMA engine.
*
* @param	IntcInstancePtr is the pointer to the INTC component instance
* @param	TxIntrId is interrupt ID associated w/ DMA TX channel
* @param	RxIntrId is interrupt ID associated w/ DMA RX channel
*
* @return	None.
*
* @note		None.
*
******************************************************************************/
static void DisableIntrSystem(INTC *IntcInstancePtr,
			      u16 TxIntrId, u16 RxIntrId)
{
	XScuGic_Disconnect(IntcInstancePtr, TxIntrId);
	XScuGic_Disconnect(IntcInstancePtr, RxIntrId);
}
