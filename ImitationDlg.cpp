// ImitationDlg.cpp : ʵ���ļ�
//
#include "stdafx.h"
#include "Imitation.h"
#include "ImitationDlg.h"
#include "afxdialogex.h"
#include <opencv.hpp>
#include <Kinect.h>
#include <thread>
#include <time.h>
#include "Motor.h"
#include <io.h> 
#include <fcntl.h>
#include <conio.h>
#include <vector>
#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;
using namespace cv;

#pragma region Global Variables
const float Alpha[7] = { 0, -pi / 2, pi / 2, -pi / 2, -pi / 2, -pi / 2, pi / 2 };
const float Theta[7] = { 0, 0, 0, -pi / 2, -pi / 2, 0, 0 };
bool Run;
double theta[7];
Motor *myMotor;

bool enableMotor = false;
bool openedMotor = false;
thread *MainThread;

int outputPeriod = 2;
int filtersize = 50;
#pragma endregion
#pragma region Global Function

void inverse_kinematics()
{

}

void draw_line(Mat& img, int* angle)
{
	int length_ = 8; 
	int startpointx = 250;
	int startpointy = 250;
	int endpointx = startpointx + int(length_*cos(angle[1] / 180.0 * 3.14159));
	int endpointy = startpointy + int(length_*sin(angle[1] / 180.0 * 3.14159));
	for (int i = 0; i < 24; i ++)
	{
		line(img, Point(startpointx, startpointy), Point(endpointx, endpointy), Scalar(255, 255, 255), -1);
		startpointx = endpointx;
		startpointy = endpointy;
		endpointx = startpointx + int(length_*cos(angle[i + 2] / 180.0 * 3.14159));
		endpointy = startpointy + int(length_*sin(angle[i + 2] / 180.0 * 3.14159));
	}
}


void Tracking() 
{
	Sleep(100);

	Mat show_result(500, 500, CV_32FC1);
	for (int i = 0; i < 500; i++)
		for (int j = 0; j < 500; j++)
			show_result.at<UINT8>(i, j) = 0;

	ifstream in("C:/Users/20600/Desktop/test.txt", ios::in);
	int angle[26];
	for (int i = 0; i < 26; i++)
		in >> angle[i];
	draw_line(show_result, angle);
	cvNamedWindow("cart", 0);
	imshow("processed cart", show_result);
	if (openedMotor && enableMotor)
		{
			double pos[6];
			pos[0] = 500; pos[1] = 0; pos[2] = -200; pos[3] = 0; pos[4] = 0; pos[5] = 0;
			myMotor->Move_J(Left, pos);
			int length_ = 5;
			int endpointx = pos[0];
			int endpointy = pos[1];
			for (int k = 1; k < 25; k++)
			{
			//int lasttime = clock();
			cvWaitKey(1);
			endpointx = endpointx + int(length_*cos(angle[k] / 180.0 * 3.14159));
			endpointy = endpointy + int(length_*sin(angle[k] / 180.0 * 3.14159));
			pos[0] = endpointx;
			pos[1] = endpointy;
			myMotor->Move_J(Left, pos);//Move(Left, theta);
			}
		}
	
}
void InitConsoleWindow()
{
	AllocConsole();
	HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
	int hCrt = _open_osfhandle((long)handle, _O_TEXT);
	FILE * hf = _fdopen(hCrt, "w");
	*stdout = *hf;
}
#pragma endregion
//-----------------------------------------

// ����Ӧ�ó��򡰹��ڡ��˵���� CAboutDlg �Ի���

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// �Ի�������
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOk();
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
	ON_BN_CLICKED(IDOK, &CAboutDlg::OnBnClickedOk)
END_MESSAGE_MAP()


// CImitationDlg �Ի���



CImitationDlg::CImitationDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_IMITATION_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CImitationDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CImitationDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CImitationDlg::OnBnClickedOk)
	ON_BN_CLICKED(openKinect, &CImitationDlg::OnBnClickedopenkinect)
	ON_BN_CLICKED(openMotor, &CImitationDlg::OnBnClickedopenmotor)
	ON_BN_CLICKED(releaseKinect, &CImitationDlg::OnBnClickedreleasekinect)
	ON_BN_CLICKED(startTracking, &CImitationDlg::OnBnClickedstarttracking)
	ON_BN_CLICKED(closeMotor, &CImitationDlg::OnBnClickedclosemotor)
	ON_BN_CLICKED(stopTracking, &CImitationDlg::OnBnClickedstoptracking)
	ON_BN_CLICKED(startImitation, &CImitationDlg::OnBnClickedstartimitation)
	ON_BN_CLICKED(stopImitation, &CImitationDlg::OnBnClickedstopimitation)
	ON_BN_CLICKED(motorHome, &CImitationDlg::OnBnClickedmotorhome)
	ON_BN_CLICKED(Photo, &CImitationDlg::OnBnClickedPhoto)
	ON_BN_CLICKED(ID_APP_ABOUT, &CImitationDlg::OnBnClickedAppAbout)
END_MESSAGE_MAP()


// CImitationDlg ��Ϣ�������

BOOL CImitationDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	InitConsoleWindow();
	_cprintf("Open console OK\n\n");
	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// ���ô˶Ի����ͼ�ꡣ  ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO: �ڴ���Ӷ���ĳ�ʼ������

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
}

void CImitationDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ  ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void CImitationDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR CImitationDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CImitationDlg::OnBnClickedOk()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	CDialogEx::OnOK();
}

void CImitationDlg::OnBnClickedopenkinect()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void CImitationDlg::OnBnClickedopenmotor()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	myMotor = new Motor(5);
	myMotor->servoOpen();
	openedMotor = true;
	_cprintf("Motor Opend!\n\n");
}


void CImitationDlg::OnBnClickedreleasekinect()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void CImitationDlg::OnBnClickedstarttracking()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	Run = true;
	MainThread = new thread(Tracking);
	_cprintf("Is Tracking now....\n\n");
}


void CImitationDlg::OnBnClickedclosemotor()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	myMotor->servoClose();
	openedMotor = false;
	_cprintf("Motor is closed!\n\n");
}


void CImitationDlg::OnBnClickedstoptracking()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	Run = false;
	_cprintf("Tracking is stopped!\n");
}


void CImitationDlg::OnBnClickedstartimitation()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	Sleep(5000);
	enableMotor = true;
	_cprintf("Motor is imitating.....\n\n");
}


void CImitationDlg::OnBnClickedstopimitation()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	enableMotor = false;
	_cprintf("Motor is not imitating any more\n\n");
}


void CImitationDlg::OnBnClickedmotorhome()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	_cprintf("Motor is going to Home position....");
	myMotor->Home(Left);
	_cprintf("Motor is Home now\n\n");
}


void CImitationDlg::OnBnClickedPhoto()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void CImitationDlg::OnBnClickedAppAbout()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
}


void CAboutDlg::OnBnClickedOk()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	CDialogEx::OnOK();
}
